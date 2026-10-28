"""Output routing, recorded-run parity and bounded evaluation for dual_output.

The scheduler imports only standard-library dependencies. Model execution and
evaluation stay in the existing experimental and original project modules.
"""

import hashlib
import math
from collections import Counter
from pathlib import Path

from fusion_ablation_utils import METRICS, MODEL_KEYS, path_key, read_csv, read_json

BASE_CONDITIONS = ("baseline", "no_TA")
CONDITIONS = (*BASE_CONDITIONS, "dual_output")
FIELDS = tuple(f"{level}-{metric}" for level in ("I", "P") for metric in METRICS)
SCORE_FIELDS = ("VA Global", "TA Global", "PQA Global", "Fixed Global Mean",
                "baseline Map Max", "baseline Image Score", "no_TA Map Max", "no_TA Image Score")


def route_outputs(maps, scores, disable=False):
    """Route [B,H,W] pixel maps and [B] image scores without recomputation.

    The image score always uses the baseline map maximum computed upstream.
    Disabling routing restores both baseline outputs. Source tensors are read
    only and shared, so there is no third full-resolution map allocation.
    """
    pixel_source = "baseline" if disable else "no_TA"
    return ({**{name: maps[name] for name in BASE_CONDITIONS},
             "dual_output": maps[pixel_source]},
            {**{name: scores[name] for name in BASE_CONDITIONS},
             "dual_output": scores["baseline"]})


def audit_ablation(source, directory):
    """Read previous full-run CSV/JSON, rejecting changed protocol or identities.

    Return finite six-metric rows and per-image score records for subsequent
    runtime parity checks. No images, matrices or weights are loaded here.
    """
    directory = Path(directory)
    status = read_json(directory / "run_status.json")
    config = read_json(directory / "experiment_config.json")
    if (status.get("status") != "completed" or status.get("complete") is not True
            or status.get("sample_count") != len(source["manifest"])
            or status.get("category_count") != len(source["classes"])):
        raise ValueError(f"Not a complete matching branch ablation: {directory}")
    for key in (*MODEL_KEYS, "k_shots", "ap_definition", "metric_scale", "map_max_stage"):
        if config.get(key) != source["config"].get(key):
            raise ValueError(f"Recorded branch ablation configuration differs: {key}")
    references = (directory / "reference_samples.txt").read_text(encoding="utf-8-sig").splitlines()
    originals = (Path(source["source"]) / "reference_samples.txt").read_text(encoding="utf-8-sig").splitlines()
    if references != originals:
        raise ValueError("Recorded branch ablation references/order differ.")
    rows = read_csv(directory / "source_sample_manifest.csv")
    scores = read_csv(directory / "image_scores.csv")
    if len(rows) != len(source["manifest"]) or len(scores) != len(rows):
        raise ValueError("Recorded branch ablation sample/score count differs.")
    for saved, score, original in zip(rows, scores, source["manifest"]):
        for row in (saved, score):
            if row["Category"] != original["Category"] or path_key(row["Image Path"]) != path_key(original["Image Path"]):
                raise ValueError("Recorded branch ablation image identity/order differs.")
        if (saved["Sample Type"] != original["Sample Type"]
                or int(score["Label"]) != int(original["Sample Type"] == "Anomaly")):
            raise ValueError("Recorded branch ablation labels differ.")
        for field in SCORE_FIELDS:
            if not math.isfinite(float(score[field])):
                raise ValueError(f"Nonfinite recorded image score: {field}")
    metrics = [row for row in read_csv(directory / "metrics_combined.csv") if row["Condition"] in BASE_CONDITIONS]
    expected = Counter((category, name) for category in [*source["classes"], "Mean"] for name in BASE_CONDITIONS)
    if Counter((row["Category"], row["Condition"]) for row in metrics) != expected:
        raise ValueError("Missing/duplicate recorded baseline or no_TA category metrics.")
    for row in metrics:
        if any(not math.isfinite(float(row[field])) or not 0 <= float(row[field]) <= 100 for field in FIELDS):
            raise ValueError("Invalid recorded branch ablation metric.")
    files = ("run_status.json", "experiment_config.json", "reference_samples.txt",
             "source_sample_manifest.csv", "image_scores.csv", "metrics_combined.csv")
    return {"directory": str(directory.resolve()), "scores": scores,
            "metrics": {(row["Category"], row["Condition"]): row for row in metrics},
            "hashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}


def compare_ablation(recorded, combined, tolerance):
    """Compare actual full-run baseline/no_TA metrics in percentage points."""
    checks = []
    for row in combined:
        if row["Condition"] not in BASE_CONDITIONS:
            continue
        old = recorded["metrics"][(row["Category"], row["Condition"])]
        for field in FIELDS:
            delta = row[field] - float(old[field]) if row[field] is not None else None
            checks.append({"Category": row["Category"], "Condition": row["Condition"], "Metric": field,
                           "Recorded": float(old[field]), "Current": row[field], "Delta": delta,
                           "Tolerance pp": tolerance, "Passed": delta is not None and abs(delta) <= tolerance})
    return checks


def compare_scores(actual, recorded, atol, rtol):
    """Summarize all eight numeric component checks for one original image."""
    deltas = {field: abs(actual[field] - float(recorded[field])) for field in SCORE_FIELDS}
    failed = [field for field in SCORE_FIELDS if not math.isfinite(actual[field])
              or not math.isclose(actual[field], float(recorded[field]), abs_tol=atol, rel_tol=rtol)]
    return {"Category": actual["Category"], "Image Path": actual["Image Path"],
            "Max Abs Delta": max(deltas.values()), "Passed": not failed, "Failed Fields": ";".join(failed)}


class CategoryBuffer:
    """Keep two unique CPU maps/score vectors per class; reuse routed metrics.

    Maps enter as [B,H,W], mask as [H,W], scores as [B]. Each original
    Evaluator is run separately per level and original condition. The routed
    metrics are copied from the exact same evaluated outputs, not estimated.
    """

    def __init__(self, category, disable=False):
        self.category, self.disable = category, disable
        self.masks, self.labels = [], []
        self.maps = {name: [] for name in BASE_CONDITIONS}
        self.scores = {name: [] for name in BASE_CONDITIONS}

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.to(torch.uint8).clone())
        self.labels.append(label)
        for name in BASE_CONDITIONS:
            self.maps[name].append(maps[name][index].clone())
            self.scores[name].append(scores[name][index].clone())

    def finish(self, device, logger):
        """Evaluate both independent outputs using unchanged repository metrics."""
        import torch
        from tools import Evaluator
        from branch_diagnostics_utils import evaluate_scores
        masks, labels = torch.stack(self.masks).int(), torch.tensor(self.labels).int()
        pixels, images = [], []
        for name in BASE_CONDITIONS:
            maps, scores = torch.stack(self.maps[name]), torch.stack(self.scores[name])
            self.maps[name].clear()
            self.scores[name].clear()
            evaluator = Evaluator(device, metrics=[f"P-{metric}" for metric in METRICS])
            pixels.append({"Category": self.category, "Condition": name, **evaluate_scores(
                evaluator, self.category, masks, maps, labels, scores, device, "pixel")})
            del evaluator, maps
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f"I-{metric}" for metric in METRICS])
            images.append({"Category": self.category, "Condition": name, **evaluate_scores(
                evaluator, self.category, dummy, dummy, labels, scores, device, "image")})
            del evaluator, dummy, scores
        pixel_source = "baseline" if self.disable else "no_TA"
        pixels.append({**next(row for row in pixels if row["Condition"] == pixel_source), "Condition": "dual_output"})
        images.append({**images[0], "Condition": "dual_output"})
        logger.info("%s: pixels=%s images=%s", self.category, pixels, images)
        self.masks.clear()
        self.labels.clear()
        return pixels, images
