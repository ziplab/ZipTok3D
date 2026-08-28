"""Evaluate prefix-length/refinement-depth grids with paper-aligned metrics."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import engine
from cod.utils.recon import (
    compute_cd_of_mesh,
    create_grid_queries,
    occupancy_to_mesh,
)
from cod.utils.training import load_model_weights_from_checkpoint


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="resolved stage-1 training config")
    parser.add_argument("checkpoint", help="stage-1 checkpoint")
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--loops", default="1,2,3,4,5,6")
    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--metrics", choices=("query", "mesh", "all"), default="all")
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--output", default="prefix_loop_metrics.csv")
    parser.add_argument("--per-object-output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=None,
                        help="optional smoke-test limit; omit for paper evaluation")
    parser.add_argument(
        "--allow-invalid-meshes", action="store_true",
        help="diagnostic only; paper metrics require a valid mesh for every object",
    )
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--cd-threshold", type=float, default=0.02)
    return parser.parse_args()


def decode_queries(model, planes, queries, chunk_size):
    chunks = []
    for start in range(0, queries.size(1), chunk_size):
        chunks.append(model.decode_queries(planes, queries[:, start:start + chunk_size]))
    return torch.cat(chunks, dim=1)


def batch_value(batch, key, default):
    value = batch.get(key, default)
    if torch.is_tensor(value):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def valid_mesh(mesh):
    return (
        mesh is not None
        and len(mesh.vertices) > 0
        and len(mesh.faces) > 0
        and np.isfinite(mesh.vertices).all()
    )


def make_mesh(logits, resolution):
    try:
        return occupancy_to_mesh(logits, resolution)
    except (RuntimeError, ValueError):
        return None


@torch.no_grad()
def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if args.max_objects is not None and args.max_objects < 1:
        raise ValueError("--max-objects must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = engine.load_config(args.config)
    model = engine.instantiate(cfg.model)
    state = load_model_weights_from_checkpoint(args.checkpoint, prefix="model")
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    tokens = parse_ints(args.tokens)
    loops = parse_ints(args.loops)
    max_tokens = model.num_latents
    max_loops = model.decoder.num_loops
    if not tokens or any(value < 1 or value > max_tokens for value in tokens):
        raise ValueError(f"token budgets must be in [1, {max_tokens}]")
    if not loops or any(value < 1 or value > max_loops for value in loops):
        raise ValueError(f"refinement depths must be in [1, {max_loops}]")
    settings = [(token_count, loop_count) for loop_count in loops for token_count in tokens]

    dm = engine.instantiate(cfg.data)
    split = args.split or dm.evaluation_split
    dataset = dm.get_dataset(split)
    dataset.repeat = 1
    dataset.use_queries = args.metrics in ("query", "all")
    dataset.use_full_surface = args.metrics in ("mesh", "all")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    grid = None
    if args.metrics in ("mesh", "all"):
        grid = create_grid_queries(dm.eval_grid_size).to(args.device)

    aggregates = {
        setting: {"iou": [], "cd": [], "f1": [], "invalid_meshes": 0}
        for setting in settings
    }
    per_object_rows = []

    for object_index, batch in enumerate(loader):
        if args.max_objects is not None and object_index >= args.max_objects:
            break
        points = batch["surface"].to(args.device, non_blocking=True)
        latent = model.encode(points)[0]
        queries = labels = None
        if args.metrics in ("query", "all"):
            queries = batch["query_points"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True).bool()

        identity = {
            "object_index": int(batch_value(batch, "idx", object_index)),
            "source": batch_value(batch, "source", ""),
            "category": batch_value(batch, "category", ""),
            "object_id": batch_value(batch, "object_id", ""),
        }

        for token_count, loop_count in settings:
            planes = model.decode(
                latent,
                active_num_latents=token_count,
                num_decode_loops=loop_count,
            )[0]
            row = {
                **identity,
                "tokens": token_count,
                "loops": loop_count,
                "query_iou": "",
                "mesh_cd": "",
                "mesh_f1": "",
                "mesh_valid": "",
            }

            if queries is not None:
                logits = decode_queries(model, planes, queries, args.chunk_size)
                predictions = logits > 0
                intersection = (predictions & labels).flatten(1).sum(dim=1).float()
                union = (predictions | labels).flatten(1).sum(dim=1).float()
                iou = (intersection / union.clamp_min(1)).item()
                aggregates[(token_count, loop_count)]["iou"].append(iou)
                row["query_iou"] = 100.0 * iou

            if grid is not None:
                grid_logits = decode_queries(model, planes, grid, args.chunk_size)
                mesh = make_mesh(grid_logits[0], dm.eval_grid_size)
                if valid_mesh(mesh):
                    np.random.seed(
                        args.seed + object_index * 1009 + token_count * 31 + loop_count
                    )
                    surface = batch["full_surface"][0].cpu().numpy()
                    cd, f1 = compute_cd_of_mesh(
                        mesh, surface, threshold=args.cd_threshold
                    )
                    aggregates[(token_count, loop_count)]["cd"].append(float(cd))
                    aggregates[(token_count, loop_count)]["f1"].append(float(f1))
                    row["mesh_cd"] = float(cd)
                    row["mesh_f1"] = 100.0 * float(f1)
                    row["mesh_valid"] = 1
                else:
                    aggregates[(token_count, loop_count)]["invalid_meshes"] += 1
                    row["mesh_valid"] = 0

            per_object_rows.append(row)
            del planes

        print(f"evaluated object {object_index + 1}/{len(dataset)}")

    invalid_settings = {
        setting: values["invalid_meshes"]
        for setting, values in aggregates.items()
        if values["invalid_meshes"]
    }
    if invalid_settings and not args.allow_invalid_meshes:
        details = ", ".join(
            f"K={tokens},L={loops}: {count}"
            for (tokens, loops), count in sorted(invalid_settings.items())
        )
        raise RuntimeError(
            "paper mesh evaluation requires one valid reconstruction per object; "
            f"invalid meshes by setting: {details}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "tokens",
        "loops",
        "query_iou",
        "mesh_cd",
        "mesh_f1",
        "num_query_objects",
        "num_valid_meshes",
        "num_invalid_meshes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for token_count, loop_count in settings:
            values = aggregates[(token_count, loop_count)]
            writer.writerow({
                "tokens": token_count,
                "loops": loop_count,
                "query_iou": (
                    100.0 * float(np.mean(values["iou"])) if values["iou"] else ""
                ),
                "mesh_cd": float(np.mean(values["cd"])) if values["cd"] else "",
                "mesh_f1": (
                    100.0 * float(np.mean(values["f1"])) if values["f1"] else ""
                ),
                "num_query_objects": len(values["iou"]),
                "num_valid_meshes": len(values["cd"]),
                "num_invalid_meshes": values["invalid_meshes"],
            })

    per_object_path = (
        Path(args.per_object_output)
        if args.per_object_output
        else output_path.with_name(f"{output_path.stem}_per_object.csv")
    )
    per_object_path.parent.mkdir(parents=True, exist_ok=True)
    object_fields = [
        "object_index",
        "source",
        "category",
        "object_id",
        "tokens",
        "loops",
        "query_iou",
        "mesh_cd",
        "mesh_f1",
        "mesh_valid",
    ]
    with per_object_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=object_fields)
        writer.writeheader()
        writer.writerows(per_object_rows)

    print(f"wrote {output_path.resolve()}")
    print(f"wrote {per_object_path.resolve()}")


if __name__ == "__main__":
    main()
