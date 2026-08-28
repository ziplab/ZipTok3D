# ZipTok3D Supplementary Code

This package contains the training, preprocessing, reconstruction, generation,
and evaluation code for **ZipTok3D: High-Fidelity 3D Tokenization with Compact
Token Prefixes**. Datasets and trained checkpoints are not included.

## Paper protocol at a glance

| Component | Paper setting |
|---|---|
| Stage-1 latent sequence | maximum 128 tokens, width 512 |
| Prefix budgets | 1, 2, 4, 8, 16, 32, 64, 128 |
| Shared decoder | 6 Transformer layers; recurrent length `192 + K` |
| Stage-1 training | 6 refinement passes; one intermediate depth sampled from 1--5 |
| Main reconstruction | 5 refinement passes |
| Prefix VAE | maximum 16 positions, width 32, 12 layers, 8 heads |
| Prefix VAE budgets | 1, 2, 4, 8, 16 with weights 1.2, 1.1, 1.0, 0.9, 0.8 |
| Stage-2 EDM training | maximum `16 x 32` array with causal self-attention |
| Reported EDM prefix | physical `2 x 32` array, without suffix padding |
| EDM | width 384, depth 16, 8 heads of width 48, causal self-attention |
| Generation sampler | 18 Heun steps, rho 7, sigma in [0.002, 80] |

The distinction between six-pass training and five-pass reporting is
intentional. The prefix VAE also uses the frozen six-pass decoder for training
supervision, while generated two-token codes are decoded with five passes.
At each refinement pass, the shared block receives the 192 recurrent selected
triplane tokens followed by the physical `K`-token prefix. Only the first 192
outputs are carried to the next pass. The other 576 initialized triplane
tokens remain outside the shared block and are restored unchanged at their
original spatial locations. There are no merged or summary tokens in the
recurrent sequence. Training examples independently sample a prefix budget;
examples with the same budget are grouped before decoding so each shared-block
call receives a physical `192 + K` sequence without suffix padding.

## Package layout

```text
cod/                         models, data modules, losses, solvers, metrics
config/                      paper training and ablation configurations
engine/                      configuration and Lightning utilities
tools/trellis500k_preprocess.py
                              TRELLIS preprocessing, splitting, and packing
tools/validate_release.py    dependency-free release audit
train_ae.py                  common training entry point for all three stages
evaluate_ae.py               one operating-point IoU/CD/F1 evaluation
evaluate_prefix_grid.py      complete K/L sweep and per-object metrics
analyze_results.py           refinement, oracle, and bootstrap diagnostics
cache_latents.py             deterministic 16-token causal-prefix cache
generate_stage2.py           class-conditioned mesh generation
evaluate_generation.py       MMD-CD, COV-CD, and 1-NNA-CD
benchmark_reconstruction.py  trained-checkpoint reconstruction efficiency
benchmark_generation.py      trained-checkpoint generation efficiency
```

`3DShape2VecSet-master/` and `3DILG-master/` are the official baseline source
snapshots used for comparison. Their licenses and original instructions remain
inside those directories. See `THIRD_PARTY.md`.

## Environment

The reported runs use Python 3.9, PyTorch 2.1.0, and CUDA 12.2. Stage-1 and
Stage-2 training use four NVIDIA A800 GPUs. Efficiency measurements use one
NVIDIA H20 GPU.

```bash
conda create -n ziptok3d python=3.9 -y
conda activate ziptok3d
pip install -r requirements.txt
git clone https://github.com/Silverster98/pointops.git ../pointops
pip install -v ../pointops
```

`pointops` provides the CUDA farthest-point sampling and KNN operators. Its
source is obtained from the upstream repository and is not redistributed in
this supplementary package.

Install TRELLIS preprocessing support separately:

```bash
pip install -r requirements-preprocessing.txt
```

W&B is disabled in every submitted configuration. To opt in, install
`requirements-wandb.txt` and supply a local config override; no private W&B
file is required for the default commands.

Run the dependency-free package audit at any time:

```bash
python tools/validate_release.py
```

## ShapeNet data

Use the 55-category ShapeNetCore-v2 records released by 3DShape2VecSet and used
by COD-VAE. Place them as follows:

```text
datasets/shapenet_vecset/
  ShapeNetV2_point/<synset>/{train,val,test}.lst
  ShapeNetV2_point/<synset>/<object>.npz
  ShapeNetV2_point/<synset>/<object>.npy
  ShapeNetV2_surface/<synset>/4_pointcloud/<object>.npz
```

The point record contains volume and near-surface occupancy queries, the
same-stem `.npy` file stores the normalization scale, and the surface record
contains the dense reference point cloud.

Convert the records to the HDF5 layout used by this code:

```bash
python preprocess_dataset.py --data shapenet
```

The paper protocol uses 48,597 training objects, a 2,592-object validation set
for checkpoint selection, and a disjoint 1,283-object test set for final
evaluation. In the released source manifests, these two held-out subsets are
physically named `test` and `val`, respectively. The data configuration maps
those source names to their paper roles. Reconstruction, ablation, K/L sweep,
and generation distribution metrics all use the 1,283-object paper test set.

## TRELLIS-500K data

The official training catalogue has 500,777 entries across Objaverse-XL
Sketchfab, Objaverse-XL GitHub, ABO, 3D-FUTURE, and HSSD. The separate Toys4K
evaluation set is not used. The effective pool consists of downloaded assets
that successfully complete polygon loading, watertight conversion,
normalization, sampling, and signed-distance evaluation. Every successful
asset is retained.

First normalize the official metadata. Repeat `--metadata` and `--asset-root`
for each available source:

```bash
python tools/trellis500k_preprocess.py manifest \
  --metadata objaverse_xl_sketchfab=<metadata.csv> \
  --asset-root objaverse_xl_sketchfab=<downloaded-assets> \
  --output datasets/trellis500k/work/manifest.jsonl
```

Process the normalized manifest. Independent ranks can run concurrently:

```bash
python tools/trellis500k_preprocess.py process \
  --manifest datasets/trellis500k/work/manifest.jsonl \
  --output-dir datasets/trellis500k/work \
  --rank 0 --world-size 1 \
  --watertight-resolution 50000 \
  --max-input-faces 50000 \
  --surface-points 100000 \
  --volume-points 500000 \
  --near-points 500000 \
  --near-stds 0.005 0.05 \
  --timeout-seconds 900 \
  --memory-limit-gb 32 \
  --seed 42
```

Common polygonal formats are read directly. Blender is optional for assets
whose container cannot be read by trimesh; install it separately and pass
`--blender <executable>`. No Blender binary is distributed with this package.

Create the paper split. The script sorts the complete successfully preprocessed
pool, applies one seed-42 permutation, assigns the first 1% to test, the next
2% to validation, and the remaining 97% to training. Counts are obtained by
rounding the requested test and validation fractions; training receives the
remainder.

The supplied 2,613-entry CSV records the source and exact object identifier of
the resulting paper test split: 57 ABO, 856 Objaverse-XL GitHub, and 1,700
Objaverse-XL Sketchfab assets. It is a verification input, not a second split
definition: the command first performs the seed-based partition and then
requires exact set equality with the CSV.

```bash
python tools/trellis500k_preprocess.py split \
  --output-dir datasets/trellis500k/work \
  --test-identifiers assets/trellis_eval_split_manifest.csv \
  --validation-fraction 0.02 \
  --test-fraction 0.01 \
  --seed 42
```

Omitting `--test-identifiers` creates the same deterministic partition but
does not verify that the available successfully preprocessed pool reproduces
the reported 2,613-object test membership. Do not omit it when reproducing the
paper split.

Pack each split into shards consumed by `config/data/trellis.yaml`:

```bash
python tools/trellis500k_preprocess.py pack \
  --split-csv datasets/trellis500k/work/splits/train.csv \
  --output-dir datasets/trellis500k/packed --split-name train
python tools/trellis500k_preprocess.py pack \
  --split-csv datasets/trellis500k/work/splits/val.csv \
  --output-dir datasets/trellis500k/packed --split-name val
python tools/trellis500k_preprocess.py pack \
  --split-csv datasets/trellis500k/work/splits/test.csv \
  --output-dir datasets/trellis500k/packed --split-name test
```

## Stage-1 tokenizer training

ShapeNet uses AdamW with learning rate `1e-4`, weight decay `0.01`, 1,000
epochs, 50 warmup epochs, and cosine decay. TRELLIS uses the same optimizer for
300 epochs at a constant learning rate. Both use per-GPU batch size 56 on four
GPUs with three-step gradient accumulation, giving global batch size 672.

```bash
python train_ae.py config/train_ziptok3d_shapenet.yaml \
  --name ziptok3d_shapenet --gpus 0,1,2,3

python train_ae.py config/train_ziptok3d_trellis.yaml \
  --name ziptok3d_trellis --gpus 0,1,2,3
```

Training uses seed 123456, FP16 mixed precision, gradient clipping at 0.5, and
selects the checkpoint with the highest mean query IoU on the 2,592-object
ShapeNet validation set or the 2% TRELLIS validation split. Each run
writes its resolved `config.yaml` and checkpoints under `logs/<name>/`.
Resume without changing the resolved protocol:

```bash
python train_ae.py logs/ziptok3d_shapenet/config.yaml \
  --name ziptok3d_shapenet_resume --resume <checkpoint.ckpt> --gpus 0,1,2,3
```

## Reconstruction evaluation

`evaluate_ae.py` uses one split for query IoU, mesh CD, and mesh F1. By default,
it reads the dataset-specific evaluation split declared in the data module:
the 1,283-object ShapeNet paper test set or the 2,613-object TRELLIS test set.
Mesh output is isolated by split, K, and L, so a later operating point cannot
reuse an earlier point's meshes. Query IoU always uses the full split. CD and
F1 require one valid extracted mesh for every object in the same split. The
paper command fails if any mesh is missing or invalid, rather than silently
changing the surface-metric denominator.

```bash
python evaluate_ae.py logs/ziptok3d_shapenet \
  --split val --tokens 1 --loops 5 --gpus '[0]'
```

Repeat with K=2 and K=4 for the main reconstruction rows. For TRELLIS, replace
the model directory with the TRELLIS run. Results are written below
`outputs/recon/val_k<K>_l<L>/` in that run directory.

The complete prefix/refinement sweep computes each object's IoU before taking
the dataset mean. It also reconstructs the `128^3` mesh and reports CD and
F1@0.02 for the same objects:

```bash
python evaluate_prefix_grid.py \
  logs/ziptok3d_shapenet/config.yaml \
  <stage1-checkpoint.ckpt> \
  --split val --metrics all \
  --tokens 1,2,4,8,16,32,64,128 \
  --loops 1,2,3,4,5,6 \
  --output results/shapenet_prefix_refinement.csv
```

The companion `*_per_object.csv` contains aligned per-object IoU/CD/F1 values
for refinement diagnostics, bootstrap analysis, and post-hoc oracle analysis.
`--max-objects` is only a smoke-test option and must be omitted for paper
numbers. `--allow-invalid-meshes` is only for diagnosing failed reconstructions;
results produced with that flag are not valid paper metrics or bootstrap input.

### Baseline operating points

The ShapeNet COD-VAE-32/64 checkpoints are inherited from the official release,
whereas COD-VAE-2 is trained using the released implementation. For TRELLIS,
COD-VAE-2/32/64 are initialized from their corresponding ShapeNet checkpoints
and fine-tuned on the same preprocessed training pool as ZipTok3D. These
checkpoints are selected by the highest mean query IoU on the same 2%
validation split. VecSet-512 uses the
inherited ShapeNet model without TRELLIS-specific training. TRELLIS results are
not reported for 3DILG or the shorter VecSet operating points. The ShapeNet
commands below reproduce the corresponding training configurations when an
official checkpoint is unavailable.

```bash
python train_ae.py config/baseline/train_codvae_k2_shapenet.yaml \
  --name codvae_k2_shapenet --gpus 0,1,2,3
python train_ae.py config/baseline/train_codvae_k32_shapenet.yaml \
  --name codvae_k32_shapenet --gpus 0,1,2,3
python train_ae.py config/baseline/train_codvae_k64_shapenet.yaml \
  --name codvae_k64_shapenet --gpus 0,1,2,3
python train_ae.py config/baseline/train_codvae_k2_trellis.yaml \
  --init-checkpoint <codvae-k2-shapenet-checkpoint.ckpt> \
  --name codvae_k2_trellis --gpus 0,1,2,3
python train_ae.py config/baseline/train_codvae_k32_trellis.yaml \
  --init-checkpoint <codvae-k32-shapenet-checkpoint.ckpt> \
  --name codvae_k32_trellis --gpus 0,1,2,3
python train_ae.py config/baseline/train_codvae_k64_trellis.yaml \
  --init-checkpoint <codvae-k64-shapenet-checkpoint.ckpt> \
  --name codvae_k64_trellis --gpus 0,1,2,3
```

Evaluate each checkpoint with its fixed token count and one decoder pass. For
example, the ShapeNet COD-VAE-32 row and its aligned per-object file are
produced by:

```bash
python evaluate_prefix_grid.py \
  logs/codvae_k32_shapenet/config.yaml \
  <codvae-k32-shapenet-checkpoint.ckpt> \
  --split val --metrics all --tokens 32 --loops 1 \
  --output results/codvae_k32_shapenet.csv
```

Run the analogous command with the K=2 and K=64 checkpoints and
configurations. For TRELLIS, use the corresponding dataset-specific run and
its exact 2,613-object test split. The COD-VAE-32 `*_per_object.csv` is the
baseline argument to the oracle and paired-bootstrap commands below.

The supplementary diagnostics consume these per-object CSVs directly:

```bash
python analyze_results.py refinement <ziptok-per-object.csv> \
  --output results/refinement_diagnostics.json
python analyze_results.py oracle \
  <ziptok-per-object.csv> <codvae-per-object.csv> \
  --output results/posthoc_oracle.json
python analyze_results.py bootstrap \
  <ziptok-per-object.csv> <codvae-per-object.csv> \
  --tokens 4 --loops 5 --expected-objects 2613 \
  --resamples 20000 --seed 123456 \
  --output results/paired_bootstrap.json
```

The paired TRELLIS bootstrap requires the same 2,613 object identifiers in
both inputs and finite IoU, CD, and F1 values for every object. Each of the
20,000 resamples draws one shared list of 2,613 indices and applies it to all
three metrics. Missing values, invalid meshes, duplicate identities, or a
different object count are fatal errors; the script never substitutes a
metric-specific valid subset.

## Mechanism ablations

The four ablation configurations correspond to the rows in the paper:

| Row | Configuration | Training/evaluation decoder |
|---|---|---|
| COD-VAE | `config/ablation/train_codvae_k2.yaml` | 12 layers x 1 pass, fixed K=2 |
| Prefix only | `config/ablation/train_prefix_only.yaml` | 12 layers x 1 pass, nested prefixes |
| Single pass | `config/ablation/train_single_pass.yaml` | 6 layers x 1 pass, nested prefixes |
| w/o intermediate supervision | `config/ablation/train_without_intermediate_supervision.yaml` | 6 layers x 5 passes, final supervision only |
| Full ZipTok3D | `config/train_ziptok3d_shapenet.yaml` | 6 layers x 6 training passes; evaluate at L=5 |

Train each configuration with `train_ae.py`, then run `evaluate_ae.py` at
`--split val --tokens 2` and the indicated pass count.

These are matched mechanism controls for the ablation table. They are distinct
from the released-protocol COD-VAE operating-point configurations under
`config/baseline/` used for the main reconstruction comparison.

## Stage-2 prefix VAE and EDM

Train the causal prefix VAE from the selected ShapeNet Stage-1 checkpoint:

```bash
python train_ae.py config/train_stage2_prefix_k16_d32.yaml \
  --name ziptok3d_prefix_vae --gpus 0,1,2,3 \
  --set solver.autoencoder_checkpoint_path=<stage1-checkpoint.ckpt>
```

This uses global batch size 376, 100 epochs, AdamW at `1e-4`, and learning-rate
halving at epochs 60, 70, 80, and 90. The selected checkpoint maximizes
query IoU on the 2,592-object ShapeNet validation set.

Cache deterministic posterior means for the full 16-position training array.
The saved arrays have physical shape `[N,16,32]`. Causal self-attention makes
every leading subsequence a valid EDM prefix.

```bash
python cache_latents.py \
  logs/ziptok3d_prefix_vae/config.yaml \
  <prefix-vae-checkpoint.ckpt> \
  latent_cache/ziptok3d_stage2_k16_d32 \
  --tokens 16 --seed 123456
```

Train the class-conditional EDM with global batch size 256:

```bash
python train_ae.py config/train_stage2_generation.yaml \
  --name ziptok3d_edm_k16_causal --gpus 0,1,2,3 \
  --set data.cache_dir=latent_cache/ziptok3d_stage2_k16_d32 \
  --set solver.normalizer_path=latent_cache/ziptok3d_stage2_k16_d32/normalizer.npz
```

The EDM runs for 1,000 epochs with 40 warmup epochs and cosine decay from
`1e-4` to `1e-6`. Every self-attention layer uses a left-to-right causal mask.
Checkpoint selection minimizes denoising loss on the same ShapeNet
model-selection split.

## Generation and generation metrics

Generate 2,000 meshes for each paper category. Category indices are airplane
0, car 16, chair 18, rifle 44, and table 49. Repeat the command for each index:

```bash
python generate_stage2.py \
  logs/ziptok3d_edm_k16_causal/config.yaml <edm-checkpoint.ckpt> \
  logs/ziptok3d_prefix_vae/config.yaml <prefix-vae-checkpoint.ckpt> \
  generated/ziptok3d_k2 \
  --category 0 --num-samples 2000 --batch-size 16 \
  --tokens 2 --num-steps 18 --loops 5
```

Evaluate all five categories:

```bash
python evaluate_generation.py generated/ziptok3d_k2 \
  --categories airplane,car,chair,table,rifle \
  --points 2048 --output results/generation_metrics.json
```

For each category, the reference set `S_r` contains every object of that
category in the 1,283-object ShapeNet paper test set. MMD-CD and COV-CD use
`5|S_r|` shapes from the deterministic 2,000-shape generated pool. 1-NNA-CD
uses `|S_r|` generated shapes and `|S_r|` references with leave-one-out self
distances excluded. Category results are averaged without weighting by
category size.

## Efficiency measurement

Efficiency commands require trained checkpoints. They do not use random
weights or synthetic shapes. Both scripts use FP32 inference, batch size 16,
disable TF32 and Flash/memory-efficient SDP, and exclude model/data loading,
marching cubes, and file output.

```bash
python benchmark_reconstruction.py \
  logs/ziptok3d_shapenet/config.yaml <stage1-checkpoint.ckpt> \
  --split val --tokens 2 --loops 5 \
  --output results/reconstruction_efficiency.json

python benchmark_generation.py \
  logs/ziptok3d_edm_k16_causal/config.yaml <edm-checkpoint.ckpt> \
  logs/ziptok3d_prefix_vae/config.yaml <prefix-vae-checkpoint.ckpt> \
  --tokens 2 --loops 5 --output results/generation_efficiency.json
```

Reconstruction uses three warmup and ten measured batches. Generation uses
five warmup and thirty measured batches. For exact paper comparisons, run on a
single NVIDIA H20 and report the detected GPU from each JSON output.

## Expected outputs and verification

- Stage-1 evaluation: `iou.json`, `cd.json`, and meshes under a unique
  `val_k<K>_l<L>` directory for ShapeNet or `test_k<K>_l<L>` for TRELLIS.
- K/L sweep: one dataset-mean CSV and one per-object CSV.
- Latent cache: train/val/test `latents.npy`, labels, `normalizer.npz`, and
  `metadata.json` recording the physical token count.
- Generation: OBJ meshes named by category and sample index.
- Generation evaluation: JSON and CSV with category rows and their mean.
- Efficiency: protocol-bearing JSON files that identify trained weights and
  the measured GPU.

Before submission or redistribution, run `python tools/validate_release.py`
and `python -m unittest discover -s tests`. The audit rejects checkpoints,
arrays, meshes, videos, caches, absolute private paths, unresolved config
inheritance, obsolete merged-token decoder settings, and a TRELLIS identifier
list whose size, uniqueness, or source composition differs from the reported
2,613-object test split.
