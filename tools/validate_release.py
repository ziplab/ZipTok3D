"""Static validation for the ZipTok3D supplementary code package."""

import ast
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "THIRD_PARTY.md",
    "train_ae.py",
    "evaluate_ae.py",
    "evaluate_prefix_grid.py",
    "analyze_results.py",
    "cache_latents.py",
    "generate_stage2.py",
    "evaluate_generation.py",
    "benchmark_reconstruction.py",
    "benchmark_generation.py",
    "tests/test_split_routing.py",
    "tests/test_decoder_protocol.py",
    "config/train_ziptok3d_shapenet.yaml",
    "config/train_ziptok3d_trellis.yaml",
    "config/train_stage2_prefix_k16_d32.yaml",
    "config/train_stage2_generation.yaml",
    "config/baseline/train_codvae_k2_shapenet.yaml",
    "config/baseline/train_codvae_k32_shapenet.yaml",
    "config/baseline/train_codvae_k64_shapenet.yaml",
    "config/baseline/train_codvae_k2_trellis.yaml",
    "config/baseline/train_codvae_k32_trellis.yaml",
    "config/baseline/train_codvae_k64_trellis.yaml",
    "assets/trellis_eval_split_manifest.csv",
)
FORBIDDEN_SUFFIXES = {
    ".ckpt", ".pt", ".pth", ".h5", ".hdf5", ".npz", ".npy",
    ".obj", ".ply", ".mp4", ".avi", ".mov", ".zip", ".tar", ".gz",
}
FORBIDDEN_TEXT = ("../logs",)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".csv"}


def fail(message):
    print(f"ERROR: {message}")
    return 1


def resolve_base(config_path, target):
    candidate = config_path.parent / target
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")
    return candidate.resolve()


def main():
    errors = 0
    vendored_pointops = ROOT / "external" / "pointops"
    if vendored_pointops.exists():
        errors += fail(
            "unlicensed pointops source must remain an external dependency"
        )
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors += fail(f"missing required file: {relative}")

    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.name.upper() == "RECOVERY.MD":
            errors += fail(f"recovery record included: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors += fail(f"data/model artifact included: {relative}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            errors += fail(f"Python cache included: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors += fail(f"non-UTF-8 text file: {relative}")
                continue
            is_validator = path.resolve() == Path(__file__).resolve()
            if not is_validator and not any(
                part in {"3DShape2VecSet-master", "3DILG-master"}
                for part in relative.parts
            ):
                for phrase in FORBIDDEN_TEXT:
                    if phrase.lower() in text.lower():
                        errors += fail(f"forbidden release text {phrase!r} in {relative}")
                if re.search(r"(?:[A-Za-z]:\\[^\\\r\n]+\\|/home/|/mnt/)", text):
                    errors += fail(f"absolute private path in {relative}")
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=str(relative))
                except SyntaxError as exc:
                    errors += fail(f"Python syntax error in {relative}: {exc}")

    config_root = ROOT / "config"
    for config_path in config_root.rglob("*.yaml"):
        text = config_path.read_text(encoding="utf-8")
        for match in re.finditer(r"^\s*_base_:\s*([^#\s]+)", text, re.MULTILINE):
            base = resolve_base(config_path, match.group(1))
            if not base.is_file():
                errors += fail(
                    f"unresolved config base {match.group(1)!r} in "
                    f"{config_path.relative_to(ROOT)}"
                )

    checks = {
        "config/_train_base.yaml": ("wandb: null",),
        "config/data/shapenet.yaml": (
            "model_selection_split: test", "evaluation_split: val",
        ),
        "config/data/latent_cache.yaml": (
            "model_selection_split: test", "evaluation_split: val",
        ),
        "config/data/trellis.yaml": (
            "model_selection_split: val", "evaluation_split: test",
        ),
        "config/model/ae_ziptok3d.yaml": (
            "num_latents: 128", "num_layers: 6", "num_loops: 6",
            "num_register: 0", "keep_ratio: 0.25",
            "nested_dropout_strategy: prefix_budget_uniform",
        ),
        "config/model/vae_prefix_k16_d32.yaml": (
            "latent_dim: 32", "num_latent_layers: 12", "active_num_latents: 16",
        ),
        "config/model/stage2_edm.yaml": (
            "num_latents: 16", "channels: 32", "width: 384", "depth: 16",
            "causal: true",
        ),
        "config/solver/ae_ziptok3d.yaml": (
            "lr: 1.0e-4", "coeff_init: 0.5", "coeff_uncertainty: 0.001",
            "elt_distill_weight: 0.5", "elt_max_loop: 6",
        ),
        "config/baseline/train_codvae_k32_shapenet.yaml": (
            "max_epochs: 100", "batch_size: 32", "num_latents: 32",
            "num_layers: 12", "num_loops: 1", "num_register: 1", "coeff_init: 1.0",
            "coeff_uncertainty: 0.01", "elt_distill: false",
        ),
    }
    for relative, fragments in checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                errors += fail(f"expected {fragment!r} in {relative}")

    source_checks = {
        "cod/data/shapenet.py": (
            "return self.eval_dataloader(self.model_selection_split)",
            "return self.eval_dataloader(self.evaluation_split)",
        ),
        "cod/data/latent_cache.py": (
            "return self.eval_dataloader(self.model_selection_split)",
            "return self.eval_dataloader(self.evaluation_split)",
        ),
        "cod/data/trellis.py": (
            "return self.eval_dataloader(self.model_selection_split)",
            "return self.eval_dataloader(self.evaluation_split)",
        ),
        "cod/models/vae/autoencoder.py": (
            "z = z[:, :active]",
            "return self._prepare_decoder_inputs(",
            "def _decode_physical_prefixes",
            "z_group = z.index_select(0, indices)[:, :prefix_length]",
        ),
        "cod/models/vae/networks/decoder.py": (
            "sequence = torch.cat([state, z], dim=1)",
            "state = updated[:, :selected.size(1)]",
            "self._restore_full_tokens(init_tokens, state, indices)",
        ),
        "evaluate_ae.py": (
            "split = args.split or dm.evaluation_split",
            "setting_name = f'{split}_k{tokens}_l{loops}'",
            "dataloader = dm.eval_dataloader(split)",
            "metric_dataset = dm.get_dataset(split)",
        ),
        "evaluate_prefix_grid.py": (
            "split = args.split or dm.evaluation_split",
            '100.0 * float(np.mean(values["iou"]))',
            '"mesh_cd"',
            '"mesh_f1"',
            'f"{output_path.stem}_per_object.csv"',
        ),
        "benchmark_reconstruction.py": (
            "split = args.split or dm.evaluation_split",
        ),
        "tools/trellis500k_preprocess.py": (
            '"--test-fraction", type=float, default=0.01',
            '"--validation-fraction", type=float, default=0.02',
            '"--test-identifiers", "--paper-eval-manifest"',
            'generated = set(test_ids)',
        ),
        "analyze_results.py": (
            '"--expected-objects", type=int, default=2613',
            '"shared_resample_indices": True',
        ),
        "evaluate_generation.py": (
            "split = args.split or dm.evaluation_split",
            '"--generated-pool-size", type=int, default=2000',
            '"generated_pool_per_category": args.generated_pool_size',
        ),
        "train_ae.py": (
            "--init-checkpoint",
            "load_model_weights_from_checkpoint(args.init_checkpoint)",
        ),
        "cache_latents.py": (
            "default=16",
            "active_num_latents=args.tokens",
            '"num_latents": args.tokens',
            '"seed": args.seed',
        ),
        "cod/models/diffusion/stage2_edm.py": (
            "causal_mask", "causal=causal", "num_latents: Optional[int] = None",
        ),
        "cod/solvers/diffusion_solver.py": (
            "EDM training expects full", "self.model.num_latents",
        ),
        "generate_stage2.py": (
            'default=2', "num_latents=args.tokens",
        ),
        "benchmark_generation.py": (
            'default=2', "num_latents=args.tokens",
        ),
    }
    for relative, fragments in source_checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                errors += fail(f"expected {fragment!r} in {relative}")

    decoder_config = (ROOT / "config/model/ae_ziptok3d.yaml").read_text(
        encoding="utf-8"
    )
    decoder_source = (ROOT / "cod/models/vae/networks/decoder.py").read_text(
        encoding="utf-8"
    )
    if "num_merged_tokens" in decoder_config:
        errors += fail("obsolete merged-token setting in ZipTok3D model config")
    if "_MergingModule" in decoder_source:
        errors += fail("obsolete merged-token module in ZipTok3D decoder")

    manifest = ROOT / "assets/trellis_eval_split_manifest.csv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 2613:
            errors += fail(f"TRELLIS evaluation manifest has {len(rows)} rows, expected 2613")
        if len({row["object_id"] for row in rows}) != len(rows):
            errors += fail("TRELLIS evaluation manifest contains duplicate object IDs")
        composition = {}
        for row in rows:
            source = row.get("category", row.get("source", ""))
            composition[source] = composition.get(source, 0) + 1
        expected_composition = {
            "abo": 57,
            "objaverse_xl_github": 856,
            "objaverse_xl_sketchfab": 1700,
        }
        if composition != expected_composition:
            errors += fail(
                "TRELLIS test identifier composition is "
                f"{composition}, expected {expected_composition}"
            )

    if errors:
        print(f"release validation failed with {errors} error(s)")
        return 1
    print(f"release validation passed: {len(files)} files, 2,613 TRELLIS test IDs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
