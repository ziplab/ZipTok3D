#!/usr/bin/env python3
"""Prepare TRELLIS-500K polygonal assets for ZipTok3D.

The pipeline deliberately separates four operations so that a multi-day run
is resumable and auditable:

1. ``manifest`` normalizes the metadata emitted by the official TRELLIS
   download toolkit.
2. ``process`` loads each polygonal asset, makes it watertight with
   point-cloud-utils, sphere-normalizes it, and samples surface/SDF points.
3. ``split`` keeps every successful item, records every failure, and creates
   deterministic train/validation/test manifests.
4. ``pack`` writes sharded HDF5 files consumed by ``cod.data.trellis``.

The geometric procedure follows the same watertight unit-sphere preprocessing
used for the ShapeNet-style SDF records. An asset is excluded only after a
concrete failure (missing file, unsupported or empty polygonal geometry,
watertight conversion failure, invalid normalized mesh, SDF failure, or
output-write failure). Failure stages and messages are preserved in
rank-specific JSONL logs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_module
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import zipfile


PAPER_TEST_SIZE = 2613
DEFAULT_SOURCES = (
    "objaverse_xl_sketchfab",
    "objaverse_xl_github",
    "abo",
    "3d_future",
    "hssd",
)
PATH_COLUMNS = ("local_path", "file_path", "path", "asset_path")
ID_COLUMNS = ("sha256", "object_id", "uid", "id")


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def canonical_source(value: str) -> str:
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "objaversexl_sketchfab": "objaverse_xl_sketchfab",
        "objaversexl_(sketchfab)": "objaverse_xl_sketchfab",
        "objaverse_xl_(sketchfab)": "objaverse_xl_sketchfab",
        "objaversexl_github": "objaverse_xl_github",
        "objaversexl_(github)": "objaverse_xl_github",
        "objaverse_xl_(github)": "objaverse_xl_github",
        "3d_future": "3d_future",
        "3dfuture": "3d_future",
    }
    return aliases.get(text, text)


def canonical_object_id(source: str, raw_id: str) -> str:
    raw = str(raw_id).strip()
    prefix = f"{source}__colon__"
    if raw.startswith(prefix):
        return raw
    if "__colon__" in raw:
        return raw
    if ":" in raw:
        left, right = raw.split(":", 1)
        return f"{canonical_source(left)}__colon__{right}"
    return f"{source}__colon__{raw}"


def parse_key_value(values: Sequence[str], option: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} expects SOURCE=PATH, got {value!r}")
        key, path = value.split("=", 1)
        result[canonical_source(key)] = path
    return result


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, records: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def first_nonempty(row: Mapping[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def command_manifest(args: argparse.Namespace) -> None:
    metadata = parse_key_value(args.metadata, "--metadata")
    roots = parse_key_value(args.asset_root, "--asset-root")
    records: List[dict] = []
    seen = set()
    source_counts: Dict[str, int] = {}

    for source, csv_name in metadata.items():
        csv_path = Path(csv_name).expanduser().resolve()
        root = Path(roots.get(source, csv_path.parent)).expanduser().resolve()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise RuntimeError(f"metadata file has no header: {csv_path}")
            for row_number, row in enumerate(reader, 2):
                raw_id = first_nonempty(row, ID_COLUMNS)
                local_path = first_nonempty(row, PATH_COLUMNS)
                if raw_id is None:
                    raise RuntimeError(
                        f"no object identifier in {csv_path}:{row_number}; "
                        f"looked for {ID_COLUMNS}"
                    )
                object_id = canonical_object_id(source, raw_id)
                if object_id in seen:
                    continue
                seen.add(object_id)
                asset_path = ""
                if local_path:
                    candidate = Path(local_path).expanduser()
                    asset_path = str(candidate if candidate.is_absolute() else root / candidate)
                records.append(
                    {
                        "item_idx": len(records),
                        "source": source,
                        "object_id": object_id,
                        "raw_id": raw_id,
                        "asset_path": asset_path,
                        "metadata_csv": str(csv_path),
                        "metadata_row": row_number,
                    }
                )
                source_counts[source] = source_counts.get(source, 0) + 1

    write_jsonl(Path(args.output), records)
    print(json.dumps({"items": len(records), "sources": source_counts}, indent=2))


def _archive_location(path_text: str) -> Optional[Tuple[Path, str]]:
    normalized = path_text.replace("\\", "/")
    lower = normalized.lower()
    suffixes = (".tar.gz/", ".tgz/", ".zip/", ".tar/")
    for marker in suffixes:
        pos = lower.find(marker)
        if pos >= 0:
            end = pos + len(marker) - 1
            return Path(normalized[:end]), normalized[end + 1 :]
    return None


@contextlib.contextmanager
def materialize_asset(path_text: str) -> Iterator[Path]:
    if not path_text:
        raise StageError("asset_missing", "metadata does not contain a local asset path")
    direct = Path(path_text).expanduser()
    if direct.is_file():
        yield direct
        return

    location = _archive_location(path_text)
    if location is None:
        raise StageError("asset_missing", f"asset does not exist: {direct}")
    archive, member = location
    if not archive.is_file():
        raise StageError("asset_missing", f"archive does not exist: {archive}")

    with tempfile.TemporaryDirectory(prefix="ziptok3d-asset-") as temp_name:
        output = Path(temp_name) / Path(member).name
        try:
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as handle, handle.open(member) as source:
                    with output.open("wb") as target:
                        shutil.copyfileobj(source, target)
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive, "r:*") as handle:
                    info = handle.getmember(member)
                    source = handle.extractfile(info)
                    if source is None:
                        raise KeyError(member)
                    with source, output.open("wb") as target:
                        shutil.copyfileobj(source, target)
            else:
                raise StageError("asset_archive", f"unsupported archive: {archive}")
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                "asset_archive", f"cannot extract {member!r} from {archive}: {exc}"
            ) from exc
        yield output


def _load_polygon_mesh(asset_path: Path, blender: Optional[str], blender_script: Path):
    import numpy as np
    import trimesh

    def load(path: Path):
        loaded = trimesh.load(str(path), process=False, skip_materials=True, force="scene")
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                raise ValueError("scene contains no geometry")
            if hasattr(loaded, "to_geometry"):
                loaded = loaded.to_geometry()
            else:
                loaded = loaded.dump(concatenate=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"expected polygonal mesh, got {type(loaded).__name__}")
        if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
            raise ValueError("mesh has no vertices or triangle faces")
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(loaded.faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("mesh is not triangulated")
        if not np.isfinite(vertices).all():
            raise ValueError("mesh has non-finite vertices")
        return vertices, faces

    try:
        return load(asset_path)
    except Exception as first_error:
        if not blender:
            raise StageError("mesh_load", str(first_error)) from first_error
        with tempfile.TemporaryDirectory(prefix="ziptok3d-blender-") as temp_name:
            converted = Path(temp_name) / "converted.ply"
            command = [
                blender,
                "--background",
                "--python",
                str(blender_script),
                "--",
                "--input",
                str(asset_path),
                "--output",
                str(converted),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0 or not converted.is_file():
                message = completed.stderr.strip() or completed.stdout.strip() or str(first_error)
                raise StageError("mesh_load", f"Blender conversion failed: {message}")
            try:
                return load(converted)
            except Exception as exc:
                raise StageError("mesh_load", f"converted mesh is invalid: {exc}") from exc


def _sample_surface(vertices, faces, count: int, rng):
    import numpy as np

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 0)
    if not valid.any():
        raise StageError("surface_sampling", "watertight mesh has no non-degenerate faces")
    face_ids = np.flatnonzero(valid)
    probabilities = areas[valid] / areas[valid].sum()
    chosen = rng.choice(face_ids, size=count, replace=True, p=probabilities)
    tri = triangles[chosen]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    return (
        (1.0 - root) * tri[:, 0]
        + root * (1.0 - uv[:, 1:]) * tri[:, 1]
        + root * uv[:, 1:] * tri[:, 2]
    )


def _sample_volume(count: int, rng):
    import numpy as np

    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    bad = norms[:, 0] == 0
    while bad.any():
        directions[bad] = rng.normal(size=(int(bad.sum()), 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        bad = norms[:, 0] == 0
    radii = np.sqrt(3.0) * np.cbrt(rng.random((count, 1)))
    return directions / norms * radii


def _signed_distance(points, vertices, faces, chunk_size: int):
    import numpy as np
    import point_cloud_utils as pcu

    outputs = []
    try:
        for start in range(0, len(points), chunk_size):
            sdf, _, _ = pcu.signed_distance_to_mesh(
                np.ascontiguousarray(points[start : start + chunk_size], dtype=np.float64),
                np.ascontiguousarray(vertices, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int32),
            )
            outputs.append(np.asarray(sdf, dtype=np.float32))
    except Exception as exc:
        raise StageError("sdf", str(exc)) from exc
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise StageError("sdf", "signed-distance query returned non-finite values")
    return result


def _process_one(record: Mapping, args: argparse.Namespace) -> dict:
    import numpy as np
    import point_cloud_utils as pcu

    started = time.time()
    source = canonical_source(str(record["source"]))
    object_id = canonical_object_id(source, str(record["object_id"]))
    digest = object_id.split("__colon__", 1)[-1]
    item_seed = (int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16) + args.seed) % (2**32)
    rng = np.random.default_rng(item_seed)
    output = Path(args.output_dir) / "processed" / source / f"{digest}.npz"
    warnings: List[str] = []

    try:
        with materialize_asset(str(record.get("asset_path", ""))) as asset_path:
            vertices, faces = _load_polygon_mesh(
                asset_path, args.blender, Path(args.blender_script).resolve()
            )
        input_faces = int(len(faces))
        decimated = False
        if args.max_input_faces > 0 and len(faces) > args.max_input_faces:
            try:
                vertices, faces, _, _ = pcu.decimate_triangle_mesh(
                    np.ascontiguousarray(vertices, dtype=np.float64),
                    np.ascontiguousarray(faces, dtype=np.int32),
                    max_faces=int(args.max_input_faces),
                )
                decimated = True
            except Exception as exc:
                # Decimation protects memory but is not itself a reason to
                # discard otherwise usable geometry. Try watertight conversion
                # on the original mesh and preserve the warning in the audit.
                warnings.append(f"decimation failed: {type(exc).__name__}: {exc}")
        try:
            watertight_vertices, watertight_faces = pcu.make_mesh_watertight(
                np.ascontiguousarray(vertices, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int32),
                int(args.watertight_resolution),
                seed=0,
            )
        except Exception as exc:
            raise StageError("watertight", str(exc)) from exc

        watertight_vertices = np.asarray(watertight_vertices, dtype=np.float64)
        watertight_faces = np.asarray(watertight_faces, dtype=np.int32)
        if len(watertight_vertices) == 0 or len(watertight_faces) == 0:
            raise StageError("watertight", "conversion returned an empty mesh")
        if not np.isfinite(watertight_vertices).all():
            raise StageError("watertight", "conversion returned non-finite vertices")

        shift = (watertight_vertices.max(axis=0) + watertight_vertices.min(axis=0)) / 2.0
        normalized = watertight_vertices - shift
        radius = np.linalg.norm(normalized, axis=1).max()
        if not np.isfinite(radius) or radius <= 0:
            raise StageError("normalization", f"invalid bounding radius {radius}")
        scale = 1.0 / radius
        normalized *= scale

        surface_points = _sample_surface(normalized, watertight_faces, args.surface_points, rng)
        volume_points = _sample_volume(args.volume_points, rng)
        volume_sdf = _signed_distance(
            volume_points, normalized, watertight_faces, args.sdf_chunk
        )

        stds = [float(x) for x in args.near_stds]
        per_std = [args.near_points // len(stds)] * len(stds)
        for index in range(args.near_points % len(stds)):
            per_std[index] += 1
        near_parts = []
        for count, std in zip(per_std, stds):
            base = _sample_surface(normalized, watertight_faces, count, rng)
            near_parts.append(base + rng.normal(scale=std, size=(count, 3)))
        near_points = np.concatenate(near_parts, axis=0)
        near_sdf = _signed_distance(near_points, normalized, watertight_faces, args.sdf_chunk)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    source=np.asarray(source),
                    object_id=np.asarray(object_id),
                    shift=shift.astype(np.float32),
                    scale=np.asarray(scale, dtype=np.float32),
                    surface_points=surface_points.astype(np.float32),
                    vol_points=volume_points.astype(np.float32),
                    vol_sdf=volume_sdf,
                    vol_label=(volume_sdf <= 0).astype(np.uint8),
                    near_points=near_points.astype(np.float32),
                    near_sdf=near_sdf,
                    near_label=(near_sdf <= 0).astype(np.uint8),
                )
            os.replace(temporary, output)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise StageError("write", str(exc)) from exc

        return {
            "status": "success",
            "stage": "complete",
            "source": source,
            "object_id": object_id,
            "asset_path": str(record.get("asset_path", "")),
            "output_path": str(output.resolve()),
            "seed": item_seed,
            "input_faces": input_faces,
            "decimated": decimated,
            "vertices": int(len(watertight_vertices)),
            "faces": int(len(watertight_faces)),
            "warnings": warnings,
            "seconds": round(time.time() - started, 3),
        }
    except StageError as exc:
        return {
            "status": "failure",
            "stage": exc.stage,
            "source": source,
            "object_id": object_id,
            "asset_path": str(record.get("asset_path", "")),
            "error_type": type(exc.__cause__ or exc).__name__,
            "error": str(exc)[:2000],
            "seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "status": "failure",
            "stage": "unexpected",
            "source": source,
            "object_id": object_id,
            "asset_path": str(record.get("asset_path", "")),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "seconds": round(time.time() - started, 3),
        }


def _isolated_worker(record: Mapping, options: Mapping, queue) -> None:
    """Run one item in a disposable process so timeouts and leaks are contained."""
    memory_limit_gb = float(options.get("memory_limit_gb", 0))
    if memory_limit_gb > 0:
        try:
            import resource

            limit = int(memory_limit_gb * (1024 ** 3))
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except ImportError:
            pass
    result = _process_one(record, argparse.Namespace(**dict(options)))
    queue.put(result)


def _process_one_with_timeout(record: Mapping, args: argparse.Namespace) -> dict:
    if args.timeout_seconds <= 0:
        return _process_one(record, args)
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else "spawn")
    queue = context.Queue(maxsize=1)
    options = {key: value for key, value in vars(args).items() if key != "func"}
    process = context.Process(target=_isolated_worker, args=(record, options, queue))
    process.start()
    process.join(args.timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join()
        return {
            "status": "failure",
            "stage": "timeout",
            "source": record["source"],
            "object_id": record["object_id"],
            "asset_path": str(record.get("asset_path", "")),
            "error_type": "TimeoutError",
            "error": f"preprocessing exceeded {args.timeout_seconds:g} seconds",
            "seconds": float(args.timeout_seconds),
        }
    try:
        return queue.get(timeout=2)
    except queue_module.Empty:
        pass
    return {
        "status": "failure",
        "stage": "worker_crash",
        "source": record["source"],
        "object_id": record["object_id"],
        "asset_path": str(record.get("asset_path", "")),
        "error_type": "ChildProcessError",
        "error": f"isolated worker exited with code {process.exitcode}",
        "seconds": 0.0,
    }


def _latest_statuses(records_dir: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for status_file in sorted(records_dir.glob("status-rank-*.jsonl")):
        for row in read_jsonl(status_file):
            latest[str(row["object_id"])] = row
    return latest


def command_process(args: argparse.Namespace) -> None:
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise SystemExit("require world_size >= 1 and 0 <= rank < world_size")
    output_dir = Path(args.output_dir).resolve()
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    status_path = records_dir / f"status-rank-{args.rank:05d}.jsonl"
    existing = _latest_statuses(records_dir) if args.resume else {}
    mode = "a" if args.resume and status_path.exists() else "w"
    selected = [
        row for index, row in enumerate(read_jsonl(Path(args.manifest)))
        if index % args.world_size == args.rank
    ]

    processed = skipped = successes = failures = 0
    with status_path.open(mode, encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(selected, 1):
            object_id = str(record["object_id"])
            previous = existing.get(object_id)
            if previous and previous.get("status") == "success" and Path(
                previous.get("output_path", "")
            ).is_file():
                skipped += 1
                continue
            if previous and previous.get("status") == "failure" and not args.retry_failures:
                skipped += 1
                continue
            result = _process_one_with_timeout(record, args)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            processed += 1
            successes += result["status"] == "success"
            failures += result["status"] == "failure"
            if index % args.log_interval == 0 or result["status"] == "failure":
                print(
                    f"rank={args.rank} seen={index}/{len(selected)} processed={processed} "
                    f"success={successes} failure={failures} skipped={skipped} "
                    f"last={object_id} stage={result['stage']}",
                    flush=True,
                )

    print(
        json.dumps(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "assigned": len(selected),
                "processed": processed,
                "successes": successes,
                "failures": failures,
                "skipped": skipped,
                "status_path": str(status_path),
            },
            indent=2,
        )
    )


def _load_test_identifiers(path: Path) -> List[str]:
    result = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            category = canonical_source(row.get("category") or row.get("source") or "")
            result.append(canonical_object_id(category, row["object_id"]))
    if len(result) != len(set(result)):
        raise RuntimeError(f"test identifier list contains duplicates: {path}")
    return result


def _write_split(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("item_idx", "source", "object_id", "npz_path")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "item_idx": index,
                    "source": row["source"],
                    "object_id": row["object_id"],
                    "npz_path": row["output_path"],
                }
            )
    os.replace(temporary, path)


def command_split(args: argparse.Namespace) -> None:
    import numpy as np

    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0, 1)")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")

    output_dir = Path(args.output_dir).resolve()
    latest = _latest_statuses(output_dir / "records")
    successes = {
        key: value for key, value in latest.items()
        if value.get("status") == "success" and Path(value.get("output_path", "")).is_file()
    }
    failure_counts: Dict[str, int] = {}
    for row in latest.values():
        if row.get("status") == "failure":
            stage = str(row.get("stage", "unknown"))
            failure_counts[stage] = failure_counts.get(stage, 0) + 1
    if not successes:
        raise RuntimeError(f"no successful preprocess records found in {output_dir / 'records'}")

    rng = np.random.default_rng(args.seed)
    shuffled = sorted(successes)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * args.test_fraction)))
    val_count = int(round(len(successes) * args.validation_fraction))
    val_count = min(val_count, len(shuffled) - test_count)
    if args.validation_fraction > 0 and val_count == 0 and len(shuffled) > test_count:
        val_count = 1

    test_ids = sorted(shuffled[:test_count])
    val_ids = sorted(shuffled[test_count : test_count + val_count])
    train_ids = sorted(shuffled[test_count + val_count :])

    identifiers_verified = False
    if args.test_identifiers:
        expected_ids = _load_test_identifiers(Path(args.test_identifiers))
        if len(expected_ids) != args.expected_test_size:
            raise RuntimeError(
                f"test identifier list has {len(expected_ids)} rows, expected "
                f"{args.expected_test_size}"
            )
        expected = set(expected_ids)
        generated = set(test_ids)
        missing = sorted(expected - generated)
        unexpected = sorted(generated - expected)
        if missing or unexpected or len(expected_ids) != len(test_ids):
            details = [
                "the seed-based test split does not match the supplied identifier list",
                f"generated={len(test_ids)}, supplied={len(expected_ids)}",
                f"missing_from_generated={len(missing)}",
                f"unexpected_in_generated={len(unexpected)}",
            ]
            if missing:
                details.append("first missing IDs: " + ", ".join(missing[:10]))
            if unexpected:
                details.append("first unexpected IDs: " + ", ".join(unexpected[:10]))
            raise RuntimeError("; ".join(details))
        identifiers_verified = True

    split_dir = output_dir / "splits"
    _write_split(split_dir / "train.csv", [successes[x] for x in train_ids])
    _write_split(split_dir / "val.csv", [successes[x] for x in val_ids])
    _write_split(split_dir / "test.csv", [successes[x] for x in test_ids])

    report = {
        "successes": len(successes),
        "failures_by_stage": failure_counts,
        "train": len(train_ids),
        "validation": len(val_ids),
        "test": len(test_ids),
        "test_identifiers_supplied": bool(args.test_identifiers),
        "test_identifiers_verified": identifiers_verified,
        "expected_test_size": args.expected_test_size,
        "requested_validation_fraction": args.validation_fraction,
        "requested_test_fraction": args.test_fraction,
        "seed": args.seed,
    }
    write_jsonl(split_dir / "split_report.jsonl", [report])
    print(json.dumps(report, indent=2))


def command_pack(args: argparse.Namespace) -> None:
    import h5py
    import numpy as np

    split_csv = Path(args.split_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with split_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index_rows = []
    split_name = args.split_name or split_csv.stem
    for shard_index, start in enumerate(range(0, len(rows), args.shard_size)):
        shard_rows = rows[start : start + args.shard_size]
        shard_name = f"{split_name}-{shard_index:05d}.h5"
        shard_path = output_dir / shard_name
        temporary = shard_path.with_suffix(".h5.tmp")
        with h5py.File(temporary, "w") as handle:
            for local_index, row in enumerate(shard_rows):
                group_name = f"{local_index:06d}"
                with np.load(row["npz_path"], allow_pickle=False) as values:
                    group = handle.create_group(group_name)
                    query_dtype = np.float16 if split_name == "train" else np.float32
                    group.create_dataset(
                        "surface_points", data=values["surface_points"].astype(np.float32),
                        compression="lzf",
                    )
                    group.create_dataset(
                        "vol_points", data=values["vol_points"].astype(query_dtype),
                        compression="lzf",
                    )
                    group.create_dataset(
                        "vol_label", data=values["vol_label"].astype(np.uint8),
                        compression="lzf",
                    )
                    group.create_dataset(
                        "near_points", data=values["near_points"].astype(query_dtype),
                        compression="lzf",
                    )
                    group.create_dataset(
                        "near_label", data=values["near_label"].astype(np.uint8),
                        compression="lzf",
                    )
                    group.attrs["source"] = row["source"]
                    group.attrs["object_id"] = row["object_id"]
                    group.attrs["scale"] = 1.0
                index_rows.append(
                    {
                        "item_idx": len(index_rows),
                        "source": row["source"],
                        "object_id": row["object_id"],
                        "shard": shard_name,
                        "group": group_name,
                    }
                )
        os.replace(temporary, shard_path)
        print(f"packed {min(start + len(shard_rows), len(rows))}/{len(rows)} -> {shard_path}")

    index_path = output_dir / f"{split_name}.csv"
    temporary = index_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("item_idx", "source", "object_id", "shard", "group")
        )
        writer.writeheader()
        writer.writerows(index_rows)
    os.replace(temporary, index_path)
    print(json.dumps({"items": len(rows), "index": str(index_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="normalize official subset metadata")
    manifest.add_argument(
        "--metadata", action="append", required=True, metavar="SOURCE=CSV",
        help="official TRELLIS metadata CSV; repeat once per subset",
    )
    manifest.add_argument(
        "--asset-root", action="append", default=[], metavar="SOURCE=DIR",
        help="root used for relative local_path values; defaults to each CSV directory",
    )
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=command_manifest)

    process = subparsers.add_parser("process", help="watertight and sample one manifest shard")
    process.add_argument("--manifest", required=True)
    process.add_argument("--output-dir", required=True)
    process.add_argument("--rank", type=int, default=0)
    process.add_argument("--world-size", type=int, default=1)
    process.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--retry-failures", action="store_true")
    process.add_argument("--blender", default=None, help="optional Blender executable for unsupported formats")
    process.add_argument(
        "--blender-script",
        default=str(Path(__file__).with_name("trellis_blender_export.py")),
    )
    process.add_argument("--watertight-resolution", type=int, default=50_000)
    process.add_argument(
        "--max-input-faces", type=int, default=50_000,
        help="best-effort decimation cap before watertight conversion; <= 0 disables it",
    )
    process.add_argument("--surface-points", type=int, default=100_000)
    process.add_argument("--volume-points", type=int, default=500_000)
    process.add_argument("--near-points", type=int, default=500_000)
    process.add_argument("--near-stds", type=float, nargs="+", default=(0.005, 0.05))
    process.add_argument("--sdf-chunk", type=int, default=100_000)
    process.add_argument(
        "--timeout-seconds", type=float, default=900.0,
        help="per-item wall-clock limit; set <= 0 to disable process isolation",
    )
    process.add_argument(
        "--memory-limit-gb", type=float, default=32.0,
        help="per-item virtual-memory limit on platforms supporting resource limits",
    )
    process.add_argument("--seed", type=int, default=42)
    process.add_argument("--log-interval", type=int, default=10)
    process.set_defaults(func=command_process)

    split = subparsers.add_parser("split", help="filter failures and create deterministic splits")
    split.add_argument("--output-dir", required=True)
    split.add_argument(
        "--test-identifiers", "--paper-eval-manifest", dest="test_identifiers",
        default=None,
        help=(
            "CSV containing the reported test identifiers; verifies the "
            "seed-based split and never changes its membership"
        ),
    )
    split.add_argument(
        "--expected-test-size", type=int, default=PAPER_TEST_SIZE,
        help="required row count when --test-identifiers is supplied",
    )
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--test-fraction", type=float, default=0.01,
        help="test fraction of the successfully preprocessed pool",
    )
    split.add_argument(
        "--validation-fraction", type=float, default=0.02,
        help="held-out validation fraction of the successful pool",
    )
    split.set_defaults(func=command_split)

    pack = subparsers.add_parser("pack", help="pack one split into sharded HDF5 files")
    pack.add_argument("--split-csv", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--split-name", choices=("train", "val", "test"), default=None)
    pack.add_argument("--shard-size", type=int, default=256)
    pack.set_defaults(func=command_pack)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
