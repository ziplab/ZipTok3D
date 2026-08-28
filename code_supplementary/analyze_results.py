"""Compute refinement, post-hoc oracle, and paired-bootstrap diagnostics."""

import argparse
import csv
import json
import math
import random
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


METRICS = ("query_iou", "mesh_cd", "mesh_f1")


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def identity(row):
    object_id = row.get("object_id", "").strip()
    source = row.get("source", "").strip()
    return f"{source}:{object_id}" if object_id else row["object_index"]


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {
                "identity": identity(raw),
                "tokens": int(raw["tokens"]),
                "loops": int(raw["loops"]),
                "mesh_valid": raw.get("mesh_valid", "1") not in {"", "0"},
            }
            for metric in METRICS:
                value = raw.get(metric, "")
                row[metric] = float(value) if value != "" else None
            rows.append(row)
    return rows


def index_setting(rows, tokens, loops):
    indexed = {}
    for row in rows:
        if row["tokens"] != tokens or row["loops"] != loops:
            continue
        key = row["identity"]
        if key in indexed:
            raise RuntimeError(
                f"duplicate object {key!r} for K={tokens}, L={loops}"
            )
        indexed[key] = row
    return indexed


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output.resolve()}")


def refinement(args):
    rows = load_rows(args.input)
    output_rows = []
    for tokens in parse_ints(args.tokens):
        for transition in args.transitions.split(","):
            source_loop, target_loop = (int(value) for value in transition.split(":"))
            source = index_setting(rows, tokens, source_loop)
            target = index_setting(rows, tokens, target_loop)
            common = sorted(set(source) & set(target))
            entry = {"tokens": tokens, "transition": f"{source_loop}->{target_loop}"}
            for metric, sign in (("query_iou", 1), ("mesh_cd", -1), ("mesh_f1", 1)):
                effects = []
                for key in common:
                    left, right = source[key], target[key]
                    if left[metric] is None or right[metric] is None:
                        continue
                    if metric != "query_iou" and not (
                        left["mesh_valid"] and right["mesh_valid"]
                    ):
                        continue
                    effects.append(sign * (right[metric] - left[metric]))
                values = effects
                if len(values) == 0:
                    raise RuntimeError(
                        f"no paired values for K={tokens}, {source_loop}->{target_loop}, {metric}"
                    )
                entry[metric] = {
                    "count": int(len(values)),
                    "improved_pct": 100.0 * sum(value > args.tolerance for value in values) / len(values),
                    "unchanged_pct": 100.0 * sum(abs(value) <= args.tolerance for value in values) / len(values),
                    "worsened_pct": 100.0 * sum(value < -args.tolerance for value in values) / len(values),
                }
            output_rows.append(entry)
    write_json(args.output, {"diagnostics": output_rows})


def round_half_up(value, places):
    quantum = Decimal("1").scaleb(-places)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def no_worse(candidate, baseline):
    if any(candidate[metric] is None or baseline[metric] is None for metric in METRICS):
        return False
    if not candidate["mesh_valid"] or not baseline["mesh_valid"]:
        return False
    return (
        round_half_up(candidate["query_iou"], 1) >= round_half_up(baseline["query_iou"], 1)
        and round_half_up(candidate["mesh_cd"], 3) <= round_half_up(baseline["mesh_cd"], 3)
        and round_half_up(candidate["mesh_f1"], 1) >= round_half_up(baseline["mesh_f1"], 1)
    )


def oracle(args):
    candidate_rows = load_rows(args.candidates)
    baseline_rows = load_rows(args.baseline)
    baseline = index_setting(baseline_rows, args.baseline_tokens, args.baseline_loops)
    by_setting = {
        (tokens, loops): index_setting(candidate_rows, tokens, loops)
        for tokens in parse_ints(args.tokens)
        for loops in parse_ints(args.loops)
    }
    selected = []
    for key in sorted(baseline):
        for tokens in sorted(parse_ints(args.tokens)):
            found = False
            for loops in sorted(parse_ints(args.loops)):
                candidate = by_setting[(tokens, loops)].get(key)
                if candidate is not None and no_worse(candidate, baseline[key]):
                    selected.append({"identity": key, **candidate})
                    found = True
                    break
            if found:
                break

    selected_ids = {row["identity"] for row in selected}
    if not selected_ids:
        raise RuntimeError("the strict oracle found no successful objects")
    baseline_subset = [baseline[key] for key in sorted(selected_ids)]
    candidate_subset = sorted(selected, key=lambda row: row["identity"])
    depth_3_or_4 = sum(row["loops"] in (3, 4) for row in candidate_subset)

    def means(rows):
        return {metric: statistics.fmean(row[metric] for row in rows) for metric in METRICS}

    payload = {
        "protocol": {
            "candidate_order": "increasing tokens, then increasing refinement depth",
            "criterion": "all metrics no worse after decimal half-up rounding",
            "rounding": {"query_iou": 1, "mesh_cd": 3, "mesh_f1": 1},
        },
        "found": len(candidate_subset),
        "total": len(baseline),
        "rate_pct": 100.0 * len(candidate_subset) / len(baseline),
        "average_tokens": statistics.fmean(row["tokens"] for row in candidate_subset),
        "average_loops": statistics.fmean(row["loops"] for row in candidate_subset),
        "three_or_four_pass_pct": 100.0 * depth_3_or_4 / len(candidate_subset),
        "candidate_subset_metrics": means(candidate_subset),
        "baseline_same_subset_metrics": means(baseline_subset),
        "selected": candidate_subset,
    }
    write_json(args.output, payload)


def bootstrap(args):
    candidate = index_setting(load_rows(args.candidates), args.tokens, args.loops)
    baseline = index_setting(
        load_rows(args.baseline), args.baseline_tokens, args.baseline_loops
    )
    common = sorted(set(candidate) & set(baseline))
    if len(common) != len(candidate) or len(common) != len(baseline):
        raise RuntimeError("bootstrap inputs must contain the same aligned objects")
    expected_objects = args.expected_objects
    if expected_objects <= 0:
        raise ValueError("--expected-objects must be positive")
    if len(common) != expected_objects:
        raise RuntimeError(
            f"bootstrap requires {expected_objects} aligned objects, got {len(common)}"
        )

    invalid = []
    for key in common:
        left, right = candidate[key], baseline[key]
        if not left["mesh_valid"] or not right["mesh_valid"]:
            invalid.append(f"{key} (invalid mesh)")
            continue
        missing = [
            metric for metric in METRICS
            if left[metric] is None or right[metric] is None
            or not math.isfinite(left[metric]) or not math.isfinite(right[metric])
        ]
        if missing:
            invalid.append(f"{key} ({', '.join(missing)})")
    if invalid:
        preview = "; ".join(invalid[:10])
        raise RuntimeError(
            "bootstrap requires finite IoU/CD/F1 pairs and valid meshes for "
            f"all {expected_objects} objects; invalid rows: {preview}"
        )

    effects = {
        metric: [candidate[key][metric] - baseline[key][metric] for key in common]
        for metric in METRICS
    }

    rng = random.Random(args.seed)
    sampled_means = {metric: [] for metric in METRICS}
    for _ in range(args.resamples):
        indices = [rng.randrange(len(common)) for _ in common]
        for metric in METRICS:
            values = effects[metric]
            sampled_means[metric].append(statistics.fmean(
                values[index] for index in indices
            ))

    def percentile(values, probability):
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    result = {}
    for metric in METRICS:
        values = sampled_means[metric]
        result[metric] = {
            "mean_effect": statistics.fmean(effects[metric]),
            "ci_2_5": percentile(values, 0.025),
            "ci_97_5": percentile(values, 0.975),
        }
    write_json(args.output, {
        "protocol": {
            "effect": "candidate minus baseline",
            "objects": len(common),
            "resamples": args.resamples,
            "seed": args.seed,
            "shared_resample_indices": True,
            "note": "negative mesh_cd and positive query_iou/mesh_f1 favor the candidate",
        },
        "effects": result,
    })


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    refine = subparsers.add_parser("refinement")
    refine.add_argument("input")
    refine.add_argument("--tokens", default="1,4")
    refine.add_argument("--transitions", default="1:3,3:5")
    refine.add_argument("--tolerance", type=float, default=1e-12)
    refine.add_argument("--output", default="refinement_diagnostics.json")
    refine.set_defaults(func=refinement)

    select = subparsers.add_parser("oracle")
    select.add_argument("candidates")
    select.add_argument("baseline")
    select.add_argument("--tokens", default="1,2,4,8,16,32")
    select.add_argument("--loops", default="1,2,3,4,5,6")
    select.add_argument("--baseline-tokens", type=int, default=32)
    select.add_argument("--baseline-loops", type=int, default=1)
    select.add_argument("--output", default="posthoc_oracle.json")
    select.set_defaults(func=oracle)

    boot = subparsers.add_parser("bootstrap")
    boot.add_argument("candidates")
    boot.add_argument("baseline")
    boot.add_argument("--tokens", type=int, default=4)
    boot.add_argument("--loops", type=int, default=5)
    boot.add_argument("--baseline-tokens", type=int, default=32)
    boot.add_argument("--baseline-loops", type=int, default=1)
    boot.add_argument("--resamples", type=int, default=20000)
    boot.add_argument("--seed", type=int, default=123456)
    boot.add_argument(
        "--expected-objects", type=int, default=2613,
        help="required number of complete aligned object pairs",
    )
    boot.add_argument("--output", default="paired_bootstrap.json")
    boot.set_defaults(func=bootstrap)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
