# ZipTok3D Supplementary Code

This repository contains the training, preprocessing, reconstruction, generation,
and evaluation code for **ZipTok3D: High-Fidelity 3D Tokenization with Compact
Token Prefixes**. Datasets and trained checkpoints are not included.

## Install

The released environment uses Python 3.9, PyTorch 2.1.0, and CUDA. From the
repository root:

```bash
conda create -n ziptok3d python=3.9 -y
conda activate ziptok3d
python -m pip install -r requirements.txt
```

Training, reconstruction, and generation evaluation also require the external
CUDA operators from `pointops`:

```bash
git clone https://github.com/Silverster98/pointops.git ../pointops
python -m pip install -v ../pointops
```

`pointops` is intentionally not redistributed here. Weights & Biases logging is
disabled by default; install the optional extra only when needed:

```bash
python -m pip install -r requirements-wandb.txt
```

## Layout

```text
cod/                         ZipTok3D models, data, losses, and solvers
config/                      Training, evaluation, and ablation configs
engine/                      Configuration and Lightning utilities
external/                    3DShape2VecSet and 3DILG source snapshots
tools/trellis500k_preprocess.py
                             TRELLIS preprocessing and split tools
train_ae.py                  Stage-1 and Stage-2 training entry point
evaluate_ae.py               Reconstruction metrics
evaluate_prefix_grid.py      Prefix/refinement sweep
generate_stage2.py           Class-conditioned mesh generation
evaluate_generation.py       Generation distribution metrics
```

The two baseline snapshots keep their original instructions and dependencies
under `external/`. See [THIRD_PARTY.md](THIRD_PARTY.md) for licenses and
upstream links.

## Data preparation

### ShapeNet

Place the 55-category ShapeNetCore-v2 records in the layout expected by
3DShape2VecSet:

```text
datasets/shapenet_vecset/
  ShapeNetV2_point/<synset>/{train,val,test}.lst
  ShapeNetV2_point/<synset>/<object>.npz
  ShapeNetV2_point/<synset>/<object>.npy
  ShapeNetV2_surface/<synset>/4_pointcloud/<object>.npz
```

Convert them to the HDF5 layout used by ZipTok3D:

```bash
python preprocess_dataset.py --data shapenet
```

### TRELLIS-500K

Run the four preprocessing stages. Repeat `--metadata` and `--asset-root` for
each available source:

```bash
python tools/trellis500k_preprocess.py manifest \
  --metadata <source>=<metadata.csv> \
  --asset-root <source>=<downloaded-assets> \
  --output datasets/trellis500k/work/manifest.jsonl

python tools/trellis500k_preprocess.py process \
  --manifest datasets/trellis500k/work/manifest.jsonl \
  --output-dir datasets/trellis500k/work

python tools/trellis500k_preprocess.py split \
  --output-dir datasets/trellis500k/work \
  --test-identifiers assets/trellis_eval_split_manifest.csv

python tools/trellis500k_preprocess.py pack \
  --split-csv datasets/trellis500k/work/splits/train.csv \
  --output-dir datasets/trellis500k/packed --split-name train
```

Run `pack` again for `val` and `test`. Blender is optional for mesh formats that
`trimesh` cannot read; pass its executable with `--blender`.

## Train and evaluate

Train the tokenizer on either dataset:

```bash
python train_ae.py config/train_ziptok3d_shapenet.yaml \
  --name ziptok3d_shapenet --gpus 0,1,2,3
python train_ae.py config/train_ziptok3d_trellis.yaml \
  --name ziptok3d_trellis --gpus 0,1,2,3
```

Evaluate a reconstruction operating point. The configured evaluation split is
used when `--split` is omitted:

```bash
python evaluate_ae.py logs/ziptok3d_shapenet \
  --tokens 2 --loops 5 --gpus '[0]'
```

For the complete prefix/refinement sweep:

```bash
python evaluate_prefix_grid.py \
  logs/ziptok3d_shapenet/config.yaml <stage1-checkpoint.ckpt> \
  --tokens 1,2,4,8,16,32,64,128 --loops 1,2,3,4,5,6 \
  --metrics all --output results/shapenet_prefix_refinement.csv
```

Stage-2 training, generation, and metric evaluation use the supplied
`config/train_stage2_*.yaml` files:

```bash
python train_ae.py config/train_stage2_prefix_k16_d32.yaml \
  --name ziptok3d_prefix_vae --gpus 0,1,2,3 \
  --set solver.autoencoder_checkpoint_path=<stage1-checkpoint.ckpt>

python cache_latents.py \
  logs/ziptok3d_prefix_vae/config.yaml <prefix-vae-checkpoint.ckpt> \
  <latent-cache> --tokens 16

python train_ae.py config/train_stage2_generation.yaml \
  --name ziptok3d_edm_k16_causal --gpus 0,1,2,3 \
  --set data.cache_dir=<latent-cache> \
  --set solver.normalizer_path=<normalizer.npz>

python generate_stage2.py \
  <generation-config> <edm-checkpoint> <vae-config> <vae-checkpoint> \
  generated/ziptok3d_k2 --category 0 --num-samples 2000 \
  --tokens 2 --loops 5

python evaluate_generation.py generated/ziptok3d_k2 \
  --categories airplane,car,chair,table,rifle
```

## Checks

Before sharing a release, run:

```bash
python tools/validate_release.py
python -m unittest discover -s tests
```

The audit rejects model/data artifacts, private absolute paths, unresolved
configuration references, and an incomplete TRELLIS evaluation manifest.
