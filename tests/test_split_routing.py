import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import trellis500k_preprocess

try:
    import numpy  # noqa: F401
except ImportError:
    NUMPY_AVAILABLE = False
else:
    NUMPY_AVAILABLE = True


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


class ShapeNetSplitRoutingTests(unittest.TestCase):
    def test_stage1_split_configuration_and_routing(self):
        config = read("config/data/shapenet.yaml")
        source = read("cod/data/shapenet.py")
        self.assertIn("model_selection_split: test", config)
        self.assertIn("evaluation_split: val", config)
        self.assertIn(
            "return self.eval_dataloader(self.model_selection_split)", source
        )
        self.assertIn("return self.eval_dataloader(self.evaluation_split)", source)

    def test_stage2_split_configuration_and_routing(self):
        config = read("config/data/latent_cache.yaml")
        source = read("cod/data/latent_cache.py")
        self.assertIn("model_selection_split: test", config)
        self.assertIn("evaluation_split: val", config)
        self.assertIn(
            "return self.eval_dataloader(self.model_selection_split)", source
        )
        self.assertIn("return self.eval_dataloader(self.evaluation_split)", source)

    def test_evaluation_defaults_to_configured_paper_test_split(self):
        source = read("evaluate_ae.py")
        self.assertIn("default=None", source)
        self.assertIn("split = args.split or dm.evaluation_split", source)
        self.assertIn("dataloader = dm.eval_dataloader(split)", source)
        self.assertIn("metric_dataset = dm.get_dataset(split)", source)

    def test_trellis_uses_two_percent_validation_and_one_percent_test(self):
        config = read("config/data/trellis.yaml")
        data_source = read("cod/data/trellis.py")
        split_source = read("tools/trellis500k_preprocess.py")
        self.assertIn("model_selection_split: val", config)
        self.assertIn("evaluation_split: test", config)
        self.assertIn(
            "return self.eval_dataloader(self.model_selection_split)", data_source
        )
        self.assertIn(
            "return self.eval_dataloader(self.evaluation_split)", data_source
        )
        self.assertIn('"--test-fraction", type=float, default=0.01', split_source)
        self.assertIn(
            '"--validation-fraction", type=float, default=0.02', split_source
        )
        self.assertIn("generated = set(test_ids)", split_source)
        self.assertIn("seed-based test split does not match", split_source)

    @unittest.skipUnless(NUMPY_AVAILABLE, "TRELLIS split execution requires NumPy")
    def test_trellis_identifier_list_verifies_but_does_not_define_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(100):
                output = root / "processed" / f"{index:03d}.npz"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                records.append({
                    "status": "success",
                    "stage": "complete",
                    "source": "abo",
                    "object_id": f"abo__colon__{index:03d}",
                    "output_path": str(output),
                })
            trellis500k_preprocess.write_jsonl(
                root / "records" / "status-rank-00000.jsonl", records
            )

            args = argparse.Namespace(
                output_dir=str(root), validation_fraction=0.02,
                test_fraction=0.01, seed=42, test_identifiers=None,
                expected_test_size=1,
            )
            trellis500k_preprocess.command_split(args)
            with (root / "splits" / "test.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                test_rows = list(csv.DictReader(stream))
            with (root / "splits" / "val.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                val_rows = list(csv.DictReader(stream))
            with (root / "splits" / "train.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                train_rows = list(csv.DictReader(stream))
            self.assertEqual((len(train_rows), len(val_rows), len(test_rows)), (97, 2, 1))

            identifiers = root / "test_identifiers.csv"
            with identifiers.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("category", "object_id"))
                writer.writeheader()
                writer.writerow({
                    "category": test_rows[0]["source"],
                    "object_id": test_rows[0]["object_id"],
                })
            args.test_identifiers = str(identifiers)
            trellis500k_preprocess.command_split(args)
            report = json.loads(
                (root / "splits" / "split_report.jsonl").read_text(encoding="utf-8")
            )
            self.assertTrue(report["test_identifiers_verified"])

            with identifiers.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("category", "object_id"))
                writer.writeheader()
                writer.writerow({
                    "category": val_rows[0]["source"],
                    "object_id": val_rows[0]["object_id"],
                })
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                trellis500k_preprocess.command_split(args)

    def test_generation_evaluation_uses_final_evaluation_split(self):
        source = read("evaluate_generation.py")
        self.assertIn("split = args.split or dm.evaluation_split", source)

    def test_weights_only_initialization_is_available_for_fine_tuning(self):
        source = read("train_ae.py")
        self.assertIn("--init-checkpoint", source)
        self.assertIn(
            "load_model_weights_from_checkpoint(args.init_checkpoint)", source
        )

    def test_all_command_runs_four_stages_and_three_packs(self):
        args = trellis500k_preprocess.build_parser().parse_args([
            "all",
            "--metadata", "abo=abo.csv",
            "--metadata", "hssd=hssd.csv",
            "--asset-root", "abo=abo-assets",
            "--work-dir", "work",
            "--packed-dir", "packed",
            "--test-identifiers", "test.csv",
        ])
        calls = []

        def record(name):
            def _call(stage_args):
                calls.append((name, stage_args))
            return _call

        with mock.patch.object(
            trellis500k_preprocess, "command_manifest", record("manifest")
        ), mock.patch.object(
            trellis500k_preprocess, "command_process", record("process")
        ), mock.patch.object(
            trellis500k_preprocess, "command_split", record("split")
        ), mock.patch.object(
            trellis500k_preprocess, "command_pack", record("pack")
        ):
            trellis500k_preprocess.command_all(args)

        self.assertEqual(
            [name for name, _ in calls],
            ["manifest", "process", "split", "pack", "pack", "pack"],
        )
        self.assertEqual(calls[0][1].metadata, ["abo=abo.csv", "hssd=hssd.csv"])
        self.assertEqual(calls[1][1].world_size, 1)
        self.assertEqual(calls[2][1].test_identifiers, "test.csv")
        self.assertEqual(
            [stage_args.split_name for name, stage_args in calls[3:]],
            ["train", "val", "test"],
        )

    def test_prepare_alias_uses_all_command(self):
        args = trellis500k_preprocess.build_parser().parse_args([
            "prepare", "--metadata", "abo=abo.csv",
        ])
        self.assertIs(args.func, trellis500k_preprocess.command_all)

    def test_all_command_stops_when_a_stage_fails(self):
        args = trellis500k_preprocess.build_parser().parse_args([
            "all", "--metadata", "abo=abo.csv",
        ])
        calls = []

        def fail_process(stage_args):
            calls.append("process")
            raise RuntimeError("preprocess failed")

        with mock.patch.object(
            trellis500k_preprocess, "command_manifest",
            lambda stage_args: calls.append("manifest"),
        ), mock.patch.object(
            trellis500k_preprocess, "command_process", fail_process
        ), mock.patch.object(
            trellis500k_preprocess, "command_split",
        ) as split, mock.patch.object(
            trellis500k_preprocess, "command_pack",
        ) as pack:
            with self.assertRaisesRegex(RuntimeError, "preprocess failed"):
                trellis500k_preprocess.command_all(args)

        self.assertEqual(calls, ["manifest", "process"])
        split.assert_not_called()
        pack.assert_not_called()

    @unittest.skipUnless(NUMPY_AVAILABLE, "TRELLIS split execution requires NumPy")
    def test_split_manifest_filters_statuses_from_older_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_dir = root / "records"
            status_rows = []
            manifest_rows = []
            for index in range(101):
                object_id = f"abo__colon__{index:03d}"
                output = root / "processed" / f"{index:03d}.npz"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                status_rows.append({
                    "status": "success",
                    "stage": "complete",
                    "source": "abo",
                    "object_id": object_id,
                    "output_path": str(output),
                })
                if index < 100:
                    manifest_rows.append({"object_id": object_id})
            trellis500k_preprocess.write_jsonl(
                records_dir / "status-rank-00000.jsonl", status_rows
            )
            manifest = root / "manifest.jsonl"
            trellis500k_preprocess.write_jsonl(manifest, manifest_rows)

            args = argparse.Namespace(
                output_dir=str(root),
                manifest=str(manifest),
                validation_fraction=0.02,
                test_fraction=0.01,
                seed=42,
                test_identifiers=None,
                expected_test_size=1,
            )
            trellis500k_preprocess.command_split(args)

            split_ids = []
            for name in ("train", "val", "test"):
                with (root / "splits" / f"{name}.csv").open(
                    "r", encoding="utf-8", newline=""
                ) as stream:
                    split_ids.extend(row["object_id"] for row in csv.DictReader(stream))
            self.assertEqual(len(split_ids), 100)
            self.assertNotIn("abo__colon__100", split_ids)


if __name__ == "__main__":
    unittest.main()
