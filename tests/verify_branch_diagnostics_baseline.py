"""Run unchanged test.py on a diagnostics smoke subset and compare actual FINAL outputs.

The Dataset subclass only selects the explicitly declared smoke subset. Reference
sampling, model initialization, inference, and Evaluator all execute inside test.py.
Original source files and existing baseline result directories are never edited.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models_test"))

import test as original
from branch_diagnostics_utils import create_run_dir, write_json


class SmokeDataset(original.Dataset):
    """Top-level definition is required by Windows DataLoader spawn/pickling."""

    smoke_per_type = 1

    def __init__(self, *dataset_args, **kwargs):
        super().__init__(*dataset_args, **kwargs)
        counts = defaultdict(int)
        selected = []
        for row in self.data_all:
            key = (row["cls_name"], row["anomaly"])
            if counts[key] < self.smoke_per_type:
                selected.append(row)
                counts[key] += 1
        self.data_all = selected
        self.length = len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic_run", type=Path, required=True)
    options = parser.parse_args()
    source = options.diagnostic_run.resolve()
    config = json.loads((source / "experiment_config.json").read_text(encoding="utf-8"))
    status = json.loads((source / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed" or config["smoke_per_type"] < 1 or not config["save_maps"]:
        parser.error("Requires a completed smoke run with --save_maps and all smoke samples selected for map saving.")
    if config["vis_per_class"] < config["smoke_per_type"] and config["maps_scope"] != "all":
        parser.error("All smoke maps are required; increase vis_per_class or use maps_scope=all.")
    output = create_run_dir(ROOT / "results" / "branch_diagnostics", f"verify_original_{source.name}")
    args = argparse.Namespace(**config)
    args.save_path = str(output)
    args.eval_metrics = config["metrics"]
    args.class_name = None
    base_dataset = original.Dataset
    base_evaluator = original.Evaluator
    captured = {}

    SmokeDataset.smoke_per_type = config["smoke_per_type"]
    class CaptureEvaluator(base_evaluator):
        def run(self, results, cls_name, logger=None):
            values = super().run(results, cls_name, logger)
            captured[cls_name] = values
            if "maps" not in captured:
                captured["maps"] = results["pr_masks"].detach().cpu().numpy()
                captured["paths"] = results["query_paths"]
            return values

    original.Dataset = SmokeDataset
    original.Evaluator = CaptureEvaluator
    try:
        original.setup_seed(args.seed)
        original.test(args)
    finally:
        original.Dataset = base_dataset
        original.Evaluator = base_evaluator
    with (source / "sample_manifest.csv").open(encoding="utf-8-sig") as stream:
        manifest = {row["Image Path"]: row for row in csv.DictReader(stream)}
    max_error = 0.0
    for path, actual in zip(captured["paths"], captured["maps"]):
        row = manifest[str(path)]
        expected = np.load(source / row["Evaluated Maps"] / "FINAL.npy", allow_pickle=False)
        max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
    metric_differences = []
    for filename, prefix in (("image_metrics.csv", "I"), ("pixel_metrics.csv", "P")):
        with (source / filename).open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["Category"] == "Mean" or row["Branch"] != "FINAL":
                    continue
                if prefix == "I" and row["Score Source"] != "official_final":
                    continue
                for key in ("AUROC", "AP", "F1max"):
                    actual = captured[row["Category"]][f"{prefix}-{key}"] * 100
                    expected = float(row[key])
                    metric_differences.append(abs(actual - expected))
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    prompts_name = f"visa_{args.seed}seed_{args.k_shots}shot_test_prompts.txt"
    original_refs = (output / Path(args.test_data_path).name / str(args.k_shots) / prompts_name).read_text().splitlines()
    diagnostic_refs = (source / prompts_name).read_text().splitlines()
    if original_refs != diagnostic_refs:
        raise AssertionError("Reference samples differ from the original test.py run.")
    report = {"status": "passed", "scope": "real_visa_smoke_subset", "diagnostic_run": str(source),
              "k_shots": args.k_shots, "sample_count": len(captured["paths"]), "reference_samples_equal": True,
              "final_maps_bitwise_equal": True, "max_final_map_abs_error": max_error,
              "max_final_metric_error_percentage_points": max(metric_differences),
              "full_dataset_fresh_baseline_compared": False,
              "note": "Executed original test.test with subset Dataset and observation-only Evaluator subclasses; original source unchanged."}
    write_json(output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
