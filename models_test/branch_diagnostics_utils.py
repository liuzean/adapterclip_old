"""Output and evaluation helpers for branch_diagnostics (no model changes)."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

BRANCHES = ("VA", "TA", "PQA", "ALIGN", "FINAL")
METRIC_NAMES = ("AUROC", "AP", "F1max")


def smooth_maps(maps, sigma):
    """Apply test.py's SciPy filter separately to each [H,W] map; return CPU [B,H,W]."""
    return torch.stack([
        torch.from_numpy(gaussian_filter(image, sigma=sigma))
        for image in maps.detach().cpu()
    ], dim=0)


def require_finite(name, value, paths):
    """Fail with sample paths instead of silently changing diagnostic scores."""
    if not torch.isfinite(value).all():
        raise ValueError(f"NaN/Inf in {name}; batch paths: {list(paths)}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, fields, rows):
    """Represent unavailable/nonfinite metrics as N/A, keeping full score precision."""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "N/A" if val is None or (
                isinstance(val, (float, np.floating)) and not np.isfinite(val)
            ) else val for key, val in row.items()})


def create_run_dir(root, name):
    """Atomically allocate a new run directory, never overwriting existing output."""
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / (name if index == 1 else f"{name}_run{index:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def sample_key(path, dataset_root):
    """Use a readable stem plus a deterministic relative-path digest to avoid collisions."""
    relative = Path(path).resolve().relative_to(Path(dataset_root).resolve()).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{Path(path).stem}_{digest}"


def save_comparison(path, image, mask, raw, evaluated):
    """Save seven aligned panels from the actual transformed model input.

    Heatmaps show evaluated (smoothed) scores with a shared Turbo color range.
    Labels give both raw and evaluated extrema. No display values return to evaluation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adaptcliplib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * np.asarray(OPENAI_DATASET_STD) + np.asarray(OPENAI_DATASET_MEAN), 0, 1)
    low = min(float(evaluated[name].min()) for name in BRANCHES)
    high = max(float(evaluated[name].max()) for name in BRANCHES)
    if high == low:
        high = low + 1e-8
    fig, axes = plt.subplots(1, 7, figsize=(22, 4.1))
    axes[0].imshow(rgb)
    axes[0].set_title("Original (model input)")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    for axis, name in zip(axes[2:], BRANCHES):
        plotted = axis.imshow(evaluated[name], cmap="turbo", vmin=low, vmax=high)
        axis.set_title(name)
        axis.set_xlabel(
            f"raw min={raw[name].min():.5g} max={raw[name].max():.5g}\n"
            f"eval min={evaluated[name].min():.5g} max={evaluated[name].max():.5g}", fontsize=8,
        )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.subplots_adjust(left=0.01, right=0.95, bottom=0.20, top=0.89, wspace=0.08)
    color_axis = fig.add_axes([0.96, 0.25, 0.008, 0.55])
    fig.colorbar(plotted, cax=color_axis, label="Smoothed anomaly score (shared scale)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_scores(evaluator, category, masks, maps, labels, scores, device, level):
    """Call the original Evaluator unmodified, omitting undefined metric cases.

    Original 'AP' is the repository's trapezoidal PR area (AUPR), deliberately
    not sklearn average_precision_score. Single-label sets are N/A for all three
    diagnostics to avoid reporting misleading degenerate scores.
    """
    prefix = "P" if level == "pixel" else "I"
    target = masks if level == "pixel" else labels
    if torch.unique(target).numel() < 2:
        return {name: None for name in METRIC_NAMES}
    results = {
        "cls_names": np.asarray([category] * len(labels)),
        "gt_masks": masks.to(device), "pr_masks": maps.to(device),
        "gt_anomalys": labels.to(device), "pr_anomalys": scores.to(device),
    }
    values = evaluator.run(results, category)
    return {name: float(values[f"{prefix}-{name}"]) * 100 for name in METRIC_NAMES}


class CategoryBuffer:
    """Keep one category on CPU; pool raw pixels for exact distribution statistics."""

    def __init__(self, category):
        self.category = category
        self.masks = []
        self.labels = []
        self.raw = {name: [] for name in BRANCHES[:-1]}
        self.maps = {name: [] for name in BRANCHES}
        self.scores = {name: [] for name in BRANCHES}

    def append(self, mask, label, raw, maps, scores):
        self.masks.append(mask)
        self.labels.append(label)
        for name in BRANCHES:
            self.maps[name].append(maps[name])
            self.scores[name].append(scores[name])
            if name != "FINAL":
                self.raw[name].append(raw[name])

    def finish(self, device, logger):
        """Return per-category pixel/image metrics and pre-smoothing pooled statistics."""
        from tools import Evaluator

        pixel_evaluator = Evaluator(device, metrics=[f"P-{name}" for name in METRIC_NAMES])
        image_evaluator = Evaluator(device, metrics=[f"I-{name}" for name in METRIC_NAMES])
        masks = torch.stack(self.masks).int()
        labels = torch.tensor(self.labels).int()
        pixel_rows, image_rows, statistics = [], [], []
        for name in BRANCHES:
            maps = torch.stack(self.maps[name])
            scores = torch.stack(self.scores[name])
            pixel_rows.append({"Category": self.category, "Branch": name, **evaluate_scores(
                pixel_evaluator, self.category, masks, maps, labels, scores, device, "pixel",
            )})
            # Image-only evaluation does not need large pixel tensors on the GPU.
            dummy = torch.zeros((len(labels), 1, 1))
            source = "map_max" if name == "ALIGN" else "native_global" if name != "FINAL" else "official_final"
            image_rows.append({
                "Category": self.category, "Branch": name, "Score Source": source,
                **evaluate_scores(image_evaluator, self.category, dummy, dummy, labels, scores, device, "image"),
            })
            if name != "ALIGN":
                image_rows.append({
                    "Category": self.category, "Branch": name, "Score Source": "map_max",
                    **evaluate_scores(image_evaluator, self.category, dummy, dummy, labels,
                                      maps.flatten(1).max(dim=1).values, device, "image"),
                })
            if name != "FINAL":
                for label, sample_type in ((0, "Normal"), (1, "Anomaly")):
                    selected = [value.numpy().ravel() for value, actual in zip(self.raw[name], self.labels) if actual == label]
                    row = {"Category": self.category, "Sample Type": sample_type, "Branch": name,
                           "Sample Count": len(selected), "Pixel Count": sum(value.size for value in selected)}
                    if selected:
                        values = np.concatenate(selected)
                        p50, p95 = np.percentile(values, [50, 95])
                        row.update(Min=float(values.min()), Max=float(values.max()),
                                   Mean=float(values.mean(dtype=np.float64)), Std=float(values.std(dtype=np.float64)),
                                   P50=float(p50), P95=float(p95))
                    else:
                        row.update({key: None for key in ("Min", "Max", "Mean", "Std", "P50", "P95")})
                    statistics.append(row)
                self.raw[name].clear()
            self.maps[name].clear()
            logger.info("Evaluated %s / %s: pixel=%s", self.category, name, pixel_rows[-1])
        return pixel_rows, image_rows, statistics


def with_macro_means(rows, grouping):
    """Average categories equally; record N/A if any category is unavailable."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in grouping)].append(row)
    output = list(rows)
    for key, group in groups.items():
        mean = {"Category": "Mean", **dict(zip(grouping, key))}
        for metric in METRIC_NAMES:
            values = [row[metric] for row in group]
            mean[metric] = float(np.mean(values)) if all(value is not None and np.isfinite(value) for value in values) else None
        output.append(mean)
    return output


def save_tables(run_dir, pixels, images, statistics):
    write_csv(run_dir / "pixel_metrics.csv", ["Category", "Branch", *METRIC_NAMES], with_macro_means(pixels, ["Branch"]))
    write_csv(run_dir / "image_metrics.csv", ["Category", "Branch", "Score Source", *METRIC_NAMES],
              with_macro_means(images, ["Branch", "Score Source"]))
    write_csv(run_dir / "branch_statistics.csv", ["Category", "Sample Type", "Branch", "Sample Count", "Pixel Count",
                                               "Min", "Max", "Mean", "Std", "P50", "P95"], statistics)


def compare_historical(run_dir, project_root, args, pixel_rows, image_rows, references):
    """Compare rounded historical tables, explicitly not claiming fresh baseline parity."""
    base = project_root / "results" / Path(args.test_data_path).name / str(args.k_shots)
    log_path = base / f"visa_{args.seed}seed_{args.k_shots}shot_test_log.txt"
    prompt_path = base / f"visa_{args.seed}seed_{args.k_shots}shot_test_prompts.txt"
    report = {"kind": "historical_rounded_metrics_only", "baseline_log": str(log_path),
              "fresh_test_py_rerun": False, "full_configuration_verified": False,
              "note": "Historical logs lack complete parameters; matching rounded metrics and references is supporting evidence only."}
    if not log_path.is_file():
        report["status"] = "unavailable"
        write_json(run_dir / "baseline_comparison.json", report)
        return
    previous = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7:
            try:
                previous[cells[0]] = [float(value) for value in cells[1:]]
            except ValueError:
                pass
    compared = []
    for pixel in pixel_rows:
        if pixel["Branch"] != "FINAL" or pixel["Category"] not in previous:
            continue
        category = pixel["Category"]
        image = next(row for row in image_rows if row["Category"] == category and row["Branch"] == "FINAL"
                     and row["Score Source"] == "official_final")
        current = [image[key] for key in METRIC_NAMES] + [pixel[key] for key in METRIC_NAMES]
        for metric, actual, expected in zip([f"{level}-{key}" for level in ("I", "P") for key in METRIC_NAMES], current, previous[category]):
            compared.append({"Category": category, "Metric": metric, "Current": actual, "Historical Rounded": expected,
                             "Within Rounding": actual is not None and abs(actual - expected) <= 0.0501})
    expected_refs = [row["relative_path"] for row in references]
    historical_refs = prompt_path.read_text(encoding="utf-8").splitlines() if prompt_path.is_file() else []
    report.update(status="compared" if compared else "no_matching_rows", comparisons=compared,
                  all_metrics_within_rounding=bool(compared) and all(row["Within Rounding"] for row in compared),
                  references_match_last_record=bool(expected_refs) and historical_refs[-len(expected_refs):] == expected_refs)
    write_json(run_dir / "baseline_comparison.json", report)
