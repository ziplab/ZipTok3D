"""Evaluate class-conditioned generation with the paper's CD-based metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm

import engine
from cod.data.shapenet import SHAPENET_CATEGORY_IDS
from pointops.functions import pointops


PAPER_CATEGORIES = ("airplane", "car", "chair", "table", "rifle")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_dir", help="directory containing generated OBJ meshes")
    parser.add_argument("--data-config", default="config/data/shapenet.yaml")
    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--categories", default=",".join(PAPER_CATEGORIES))
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument(
        "--generated-pool-size", type=int, default=2000,
        help="number of generated candidates per category before metric subsampling",
    )
    parser.add_argument("--pair-batch", type=int, default=32,
                        help="maximum number of shape pairs in one CUDA KNN call")
    parser.add_argument("--output", default="generation_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123456)
    return parser.parse_args()


def load_category_map(path):
    with open(path, "r", encoding="utf-8") as stream:
        names = json.load(stream)
    return {name: SHAPENET_CATEGORY_IDS.index(synset) for synset, name in names.items()}


def as_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values()
                  if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"no polygonal mesh in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"invalid mesh: {path}")
    return loaded


def sample_mesh(path, count, seed):
    mesh = as_mesh(path)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points = mesh.sample(count)
    finally:
        np.random.set_state(state)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f"invalid sampled surface: {path}")
    return torch.from_numpy(points.astype(np.float32))


def sample_reference(points, count, seed):
    generator = np.random.default_rng(seed)
    replace = len(points) < count
    indices = generator.choice(len(points), size=count, replace=replace)
    return torch.from_numpy(np.asarray(points[indices], dtype=np.float32))


def generated_meshes(root, category_index):
    root = Path(root)
    candidates = set(root.glob(f"category{category_index:02d}_*.obj"))
    candidates.update((root / f"category{category_index:02d}").glob("*.obj"))
    return sorted(candidates)


@torch.no_grad()
def paired_chamfer(left, right):
    if left.shape != right.shape or left.ndim != 3 or left.size(-1) != 3:
        raise ValueError("paired point clouds must have equal [B, N, 3] shapes")
    _, left_to_right = pointops.knn(left.contiguous(), right.contiguous(), 1)
    _, right_to_left = pointops.knn(right.contiguous(), left.contiguous(), 1)
    return left_to_right.squeeze(-1).mean(1) + right_to_left.squeeze(-1).mean(1)


@torch.no_grad()
def pairwise_chamfer(left, right, device, pair_batch, description):
    if pair_batch < 1:
        raise ValueError("--pair-batch must be positive")
    result = np.empty((len(left), len(right)), dtype=np.float32)
    pairs = [(i, j) for i in range(len(left)) for j in range(len(right))]
    for start in tqdm(range(0, len(pairs), pair_batch), desc=description):
        selected = pairs[start:start + pair_batch]
        left_batch = torch.stack([left[i] for i, _ in selected]).to(device)
        right_batch = torch.stack([right[j] for _, j in selected]).to(device)
        distances = paired_chamfer(left_batch, right_batch).cpu().numpy()
        for (i, j), distance in zip(selected, distances):
            result[i, j] = distance
    return result


def category_metrics(reference, generated, device, pair_batch, name):
    reference_count = len(reference)
    distribution_generated = generated[:5 * reference_count]
    nna_generated = generated[:reference_count]

    reference_to_generated = pairwise_chamfer(
        reference, distribution_generated, device, pair_batch, f"{name}: MMD/COV"
    )
    mmd = float(reference_to_generated.min(axis=1).mean())
    covered = np.unique(reference_to_generated.argmin(axis=0)).size
    coverage = 100.0 * covered / reference_count

    ref_ref = pairwise_chamfer(
        reference, reference, device, pair_batch, f"{name}: reference NNA"
    )
    gen_gen = pairwise_chamfer(
        nna_generated, nna_generated, device, pair_batch, f"{name}: generated NNA"
    )
    ref_gen = reference_to_generated[:, :reference_count]
    np.fill_diagonal(ref_ref, np.inf)
    np.fill_diagonal(gen_gen, np.inf)
    reference_correct = ref_ref.min(axis=1) < ref_gen.min(axis=1)
    generated_correct = gen_gen.min(axis=1) < ref_gen.min(axis=0)
    nna = 100.0 * float(
        (reference_correct.sum() + generated_correct.sum()) / (2 * reference_count)
    )
    return {"mmd_cd": mmd, "cov_cd": coverage, "1_nna_cd": nna}


def main():
    args = parse_args()
    if args.points != 2048:
        print("warning: the paper protocol uses exactly 2,048 points per shape")
    if args.generated_pool_size < 1:
        raise ValueError("--generated-pool-size must be positive")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("generation metrics require the compiled CUDA pointops KNN")

    category_names = [item.strip() for item in args.categories.split(",") if item.strip()]
    name_to_index = load_category_map("assets/shapenet_synset_dict.json")
    unknown = sorted(set(category_names) - set(name_to_index))
    if unknown:
        raise ValueError(f"unknown ShapeNet categories: {unknown}")

    cfg = engine.load_config(args.data_config)
    dm = engine.instantiate(cfg)
    split = args.split or dm.evaluation_split
    dataset = dm.get_dataset(split)
    dataset.use_queries = False
    dataset.use_full_surface = True

    references = {name: [] for name in category_names}
    indices_to_names = {name_to_index[name]: name for name in category_names}
    for item_index in tqdm(range(len(dataset)), desc=f"load {split} references"):
        item = dataset[item_index]
        category_name = indices_to_names.get(int(item["category_ids"]))
        if category_name is None:
            continue
        references[category_name].append(sample_reference(
            item["full_surface"].numpy(), args.points, args.seed + item_index
        ))

    rows = []
    for category_offset, category_name in enumerate(category_names):
        category_index = name_to_index[category_name]
        reference = references[category_name]
        paths = generated_meshes(args.generated_dir, category_index)
        if len(paths) < args.generated_pool_size:
            raise RuntimeError(
                f"{category_name}: found {len(paths)} generated meshes, need the "
                f"complete {args.generated_pool_size}-shape candidate pool"
            )
        candidate_pool = paths[:args.generated_pool_size]
        required = 5 * len(reference)
        if len(candidate_pool) < required:
            raise RuntimeError(
                f"{category_name}: the {args.generated_pool_size}-shape candidate "
                f"pool is smaller than 5|S_r|={required} for |S_r|={len(reference)}"
            )
        generated = [
            sample_mesh(path, args.points, args.seed + category_offset * 100000 + index)
            for index, path in enumerate(tqdm(
                candidate_pool[:required], desc=f"sample {category_name}"
            ))
        ]
        metrics = category_metrics(
            reference, generated, args.device, args.pair_batch, category_name
        )
        rows.append({
            "category": category_name,
            "category_index": category_index,
            "reference_count": len(reference),
            "generated_count_mmd_cov": required,
            "generated_count_1_nna": len(reference),
            **metrics,
        })

    mean_row = {
        "category": "mean",
        "category_index": "",
        "reference_count": sum(row["reference_count"] for row in rows),
        "generated_count_mmd_cov": sum(row["generated_count_mmd_cov"] for row in rows),
        "generated_count_1_nna": sum(row["generated_count_1_nna"] for row in rows),
        "mmd_cd": float(np.mean([row["mmd_cd"] for row in rows])),
        "cov_cd": float(np.mean([row["cov_cd"] for row in rows])),
        "1_nna_cd": float(np.mean([row["1_nna_cd"] for row in rows])),
    }
    rows.append(mean_row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "split": split,
            "categories": category_names,
            "points_per_shape": args.points,
            "generated_pool_per_category": args.generated_pool_size,
            "chamfer": "sum of bidirectional mean Euclidean nearest-neighbor distances",
            "mmd_cov_generated_ratio": 5,
            "1_nna_sets": "equal size with leave-one-out self distances excluded",
            "seed": args.seed,
        },
        "results": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output.resolve()}")
    print(f"wrote {csv_output.resolve()}")


if __name__ == "__main__":
    main()
