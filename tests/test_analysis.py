import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_results


FIELDS = [
    "object_index", "source", "category", "object_id", "tokens", "loops",
    "query_iou", "mesh_cd", "mesh_f1", "mesh_valid",
]


class AnalysisTests(unittest.TestCase):
    def write_rows(self, directory, name, rows):
        path = Path(directory) / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def row(index, tokens, loops, iou, cd, f1):
        return {
            "object_index": index,
            "source": "test",
            "category": "",
            "object_id": f"object-{index}",
            "tokens": tokens,
            "loops": loops,
            "query_iou": iou,
            "mesh_cd": cd,
            "mesh_f1": f1,
            "mesh_valid": 1,
        }

    def test_refinement_and_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = []
            baseline = []
            for index in (0, 1):
                candidates.extend([
                    self.row(index, 1, 1, 90.0, 0.020, 90.0),
                    self.row(index, 1, 3, 97.05, 0.0114, 98.05),
                    self.row(index, 1, 5, 97.2, 0.0110, 98.2),
                ])
                baseline.append(self.row(index, 32, 1, 97.0, 0.011, 98.0))
            candidate_path = self.write_rows(directory, "candidates.csv", candidates)
            baseline_path = self.write_rows(directory, "baseline.csv", baseline)

            refinement_output = Path(directory) / "refinement.json"
            analyze_results.refinement(argparse.Namespace(
                input=str(candidate_path), tokens="1", transitions="1:3,3:5",
                tolerance=1e-12, output=str(refinement_output),
            ))
            diagnostics = json.loads(refinement_output.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["diagnostics"][0]["query_iou"]["improved_pct"], 100.0)

            oracle_output = Path(directory) / "oracle.json"
            analyze_results.oracle(argparse.Namespace(
                candidates=str(candidate_path), baseline=str(baseline_path),
                tokens="1", loops="1,3,5", baseline_tokens=32,
                baseline_loops=1, output=str(oracle_output),
            ))
            result = json.loads(oracle_output.read_text(encoding="utf-8"))
            self.assertEqual(result["found"], 2)
            self.assertEqual(result["average_tokens"], 1.0)
            self.assertEqual(result["average_loops"], 3.0)

    def test_bootstrap_zero_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = [self.row(index, 4, 5, 97.0, 0.011, 98.0) for index in (0, 1)]
            baseline = [self.row(index, 32, 1, 97.0, 0.011, 98.0) for index in (0, 1)]
            candidate_path = self.write_rows(directory, "candidates.csv", candidates)
            baseline_path = self.write_rows(directory, "baseline.csv", baseline)
            output = Path(directory) / "bootstrap.json"
            analyze_results.bootstrap(argparse.Namespace(
                candidates=str(candidate_path), baseline=str(baseline_path),
                tokens=4, loops=5, baseline_tokens=32, baseline_loops=1,
                resamples=20, seed=123456, expected_objects=2,
                output=str(output),
            ))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["effects"]["mesh_cd"]["mean_effect"], 0.0)
            self.assertEqual(result["protocol"]["objects"], 2)
            self.assertTrue(result["protocol"]["shared_resample_indices"])

    def test_bootstrap_rejects_incomplete_metric_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = [
                self.row(0, 4, 5, 97.0, 0.011, 98.0),
                self.row(1, 4, 5, 96.0, "", ""),
            ]
            candidates[1]["mesh_valid"] = 0
            baseline = [
                self.row(0, 32, 1, 96.5, 0.012, 97.5),
                self.row(1, 32, 1, 95.5, 0.013, 97.0),
            ]
            candidate_path = self.write_rows(directory, "candidates.csv", candidates)
            baseline_path = self.write_rows(directory, "baseline.csv", baseline)
            output = Path(directory) / "bootstrap.json"
            with self.assertRaisesRegex(RuntimeError, "finite IoU/CD/F1 pairs"):
                analyze_results.bootstrap(argparse.Namespace(
                    candidates=str(candidate_path), baseline=str(baseline_path),
                    tokens=4, loops=5, baseline_tokens=32, baseline_loops=1,
                    resamples=20, seed=123456, expected_objects=2,
                    output=str(output),
                ))

    def test_bootstrap_rejects_wrong_object_count(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates = [self.row(0, 4, 5, 97.0, 0.011, 98.0)]
            baseline = [self.row(0, 32, 1, 96.5, 0.012, 97.5)]
            candidate_path = self.write_rows(directory, "candidates.csv", candidates)
            baseline_path = self.write_rows(directory, "baseline.csv", baseline)
            with self.assertRaisesRegex(RuntimeError, "requires 2613 aligned objects"):
                analyze_results.bootstrap(argparse.Namespace(
                    candidates=str(candidate_path), baseline=str(baseline_path),
                    tokens=4, loops=5, baseline_tokens=32, baseline_loops=1,
                    resamples=20, seed=123456, expected_objects=2613,
                    output=str(Path(directory) / "bootstrap.json"),
                ))


if __name__ == "__main__":
    unittest.main()
