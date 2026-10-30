"""I/O, recorded-reference audit and diagnostics for fusion_ablation.

Top-level imports are standard library only. The parent scheduler never imports
PyTorch or initializes CUDA. Numerical helpers import dependencies when called.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = ("baseline", "no_align", "pqa_align")
METRICS = ("AUROC", "AP", "F1max")
MODEL_KEYS = (
    "dataset", "test_data_path", "pretrained_model", "checkpoint_path",
    "features_list", "batch_size", "image_size", "n_ctx", "seed", "sigma",
    "fusion_type", "vl_reduction", "pq_mid_dim", "pq_context",
    "visual_learner", "textual_learner", "pq_learner",
)
STAT_FIELDS = (
    "Shot", "Category", "Sample Type", "Image Path", "Map", "Stage", "Dtype",
    "Height", "Width", "Min", "Max", "Mean", "Std", "P50", "P95",
    "Foreground Mean", "Foreground Max", "Foreground P95", "Background Mean",
    "Background P95", "Foreground Minus Background", "Top 1% Mean",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "N/A" if value is None or (
                isinstance(value, float) and not math.isfinite(value)) else value
                for key, value in row.items()})


def new_directory(root, name):
    """Allocate a fresh numbered run; existing results are never reused."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = root / (name if index == 1 else f"{name}_run{index:02d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def project_path(value, project_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(project_root) / path).resolve()


def path_key(value):
    return str(Path(value).resolve()).casefold()


def audit_source(source, shot, project_root, expected_dataset="visa"):
    """Read metadata/CSV only; return config, exact reference rows and manifest.

    References are looked up in TRAIN metadata without random sampling. Duplicate
    references, if originally sampled with replacement, retain their multiplicity.
    Test samples must match metadata in order, category and label.
    """
    source = Path(source)
    config = read_json(source / "experiment_config.json")
    status = read_json(source / "run_status.json")
    required = set(MODEL_KEYS) | {"k_shots"}
    if required - config.keys():
        raise ValueError(f"Missing config fields: {sorted(required - config.keys())}")
    if expected_dataset not in ("visa", "btad", "mpdd", "SDD", "DTD", "part_glass"):
        raise ValueError(f"Unsupported diagnostic dataset: {expected_dataset}")
    if (config["dataset"], config["seed"], config["k_shots"]) != (expected_dataset, 10, shot):
        raise ValueError(f"Unexpected dataset/seed/shot in {source}")
    if config["features_list"] != [6, 12, 18, 24] or config["fusion_type"] != "average_mean":
        raise ValueError("This predefined experiment requires layers 6/12/18/24 and average_mean.")
    if not all(config[key] is True for key in ("visual_learner", "textual_learner", "pq_learner", "pq_context")):
        raise ValueError("All three adapters and PQContext must be enabled.")
    expected_scope = "full_visa" if expected_dataset == "visa" else "full_dataset"
    if (status.get("status") != "completed" or status.get("evaluation_scope") != expected_scope
            or config.get("smoke_per_type", 0)):
        raise ValueError(f"Source is not a completed full {expected_dataset} run: {source}")
    for key in ("test_data_path", "pretrained_model", "checkpoint_path"):
        config[key] = str(project_path(config[key], project_root))
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])
    root = Path(config["test_data_path"])
    metadata = read_json(root / "meta.json")
    classes = list(metadata["test"])
    if expected_dataset == "visa" and len(classes) != 12:
        raise ValueError("Expected 12 VisA classes.")
    if not classes:
        raise ValueError("Empty test class list.")
    train = {path_key(root / row["img_path"]): row
             for rows in metadata["train"].values() for row in rows}
    references, reference_paths = [], []
    for line in (source / "reference_samples.txt").read_text(encoding="utf-8-sig").splitlines():
        category, value = line.split("\t", 1)
        path = project_path(value, root)
        path.relative_to(root)  # reject references outside the configured dataset
        row = train.get(path_key(path))
        if row is None or row["cls_name"] != category or int(row["anomaly"]) != 0:
            raise ValueError(f"Reference must be a normal TRAIN image of {category}: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        references.append(dict(row))
        reference_paths.append((category, str(path)))
    if Counter(category for category, _ in reference_paths) != Counter({c: shot for c in classes}):
        raise ValueError("Recorded reference counts do not match shot configuration.")
    manifest = read_csv(source / "sample_manifest.csv")
    test_rows = [row for category in classes for row in metadata["test"][category]]
    if len(manifest) != len(test_rows) or len(manifest) != status.get("sample_count"):
        raise ValueError("Source manifest/metadata/status sample counts differ.")
    seen = set()
    for saved, row in zip(manifest, test_rows):
        path = root / row["img_path"]
        key = path_key(path)
        if (saved["Category"] != row["cls_name"] or path_key(saved["Image Path"]) != key
                or saved["Sample Type"] != ("Anomaly" if int(row["anomaly"]) else "Normal")):
            raise ValueError(f"Test sample/order/label mismatch: {path}")
        if key in seen:
            raise ValueError(f"Duplicate test image: {path}")
        seen.add(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(row["anomaly"]) and not (root / row["mask_path"]).is_file():
            raise FileNotFoundError(root / row["mask_path"])
    diagnostics = []
    if expected_dataset == "visa":
        for stem, label in (("000", "Anomaly"), ("001", "Anomaly"), ("002", "Anomaly"), ("0790", "Normal")):
            matches = [r for r in manifest if r["Category"] == "macaroni1"
                       and r["Sample Type"] == label and Path(r["Image Path"]).stem == stem]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely locate macaroni1 {label}/{stem} in source manifest.")
            diagnostics.append(matches[0])
    else:
        # Same four-image diagnostic budget as fusion_ablation, with no VisA
        # category/stem assumptions or selection based on prediction quality.
        for label, count in (("Anomaly", 3), ("Normal", 1)):
            candidates = [r for r in manifest if r["Category"] == classes[0] and r["Sample Type"] == label]
            if len(candidates) < count:
                raise ValueError(f"Insufficient fixed {label} diagnostics in {classes[0]}.")
            diagnostics.extend(candidates[:count])
    metrics = {}
    for level, filename in (("I", "image_metrics.csv"), ("P", "pixel_metrics.csv")):
        rows = [r for r in read_csv(source / filename) if r["Branch"] == "FINAL"
                and (level == "P" or r.get("Score Source") == "official_final")]
        if Counter(r["Category"] for r in rows) != Counter(classes + ["Mean"]):
            raise ValueError(f"Incomplete/duplicate FINAL metrics in {filename}")
        for row in rows:
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f"Invalid recorded metric: {row}")
                metrics[(row["Category"], f"{level}-{metric}")] = value
    # Read statistics as part of the source audit; no recomputation or modification.
    statistics = read_csv(source / "branch_statistics.csv")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in (
        "experiment_config.json", "reference_samples.txt", "sample_manifest.csv",
        "image_metrics.csv", "pixel_metrics.csv", "branch_statistics.csv")}
    return dict(source=str(source.resolve()), config=config, classes=classes,
                references=references, reference_paths=reference_paths, manifest=manifest,
                diagnostics=diagnostics, metrics=metrics, hashes=hashes,
                statistics_rows=len(statistics))


def audit_sequence(sources):
    """Check model settings and query order agree; report nested reference lists."""
    first = sources[0]
    differences = []
    for other in sources[1:]:
        differences.extend(key for key in MODEL_KEYS if first["config"][key] != other["config"][key])
        differences.extend(key for key in ("metrics", "metric_scale", "map_max_stage", "ap_definition", "versions")
                           if first["config"].get(key) != other["config"].get(key))
        if [(r["Category"], path_key(r["Image Path"])) for r in first["manifest"]] != [
                (r["Category"], path_key(r["Image Path"])) for r in other["manifest"]]:
            differences.append("test sample order")
    if differences:
        raise ValueError(f"Cross-shot configuration mismatch: {sorted(set(differences))}")
    nested = []
    for low, high in zip(sources, sources[1:]):
        by_low, by_high = defaultdict(list), defaultdict(list)
        for category, path in low["reference_paths"]:
            by_low[category].append(path_key(path))
        for category, path in high["reference_paths"]:
            by_high[category].append(path_key(path))
        nested.append({"from": low["config"]["k_shots"], "to": high["config"]["k_shots"],
                       "ordered_prefix": all(paths == by_high[c][:len(paths)] for c, paths in by_low.items())})
    return {"model_config_equal": True, "test_order_equal": True, "reference_nesting": nested,
            "runs": [{"shot": s["config"]["k_shots"], "source": s["source"],
                      "sample_count": len(s["manifest"]), "class_count": len(s["classes"]),
                      "reference_count": len(s["references"]), "normal_train_references": True,
                      "statistics_rows": s["statistics_rows"], "source_file_sha256": s["hashes"],
                      "recorded_baseline_mean": {metric: s["metrics"][("Mean", metric)]
                                                 for metric in [f"{l}-{m}" for l in ("I", "P") for m in METRICS]}}
                     for s in sources],
            "weights": "Paths and existence checked; historical weight contents cannot be established from old configs.",
            "metric_definition": "Original tools.Evaluator; AP is repository trapezoidal PR area; percent scale.",
            "old_report_discrepancies": "Use source CSVs. Unmatched earlier numbers remain unverified; do not infer their origin."}


class RecordedPromptDataset:
    """Reuse original PromptDataset.__getitem__ without invoking its sampler."""

    def __init__(self, root, rows, classes, transform, target_transform):
        self.root, self.prompt_data_all = root, rows
        self.transform, self.target_transform = transform, target_transform
        self.view_list = ["0"]
        self.class_name_map_class_id = {name: index for index, name in enumerate(classes)}

    def __len__(self):
        return len(self.prompt_data_all)

    def __getitem__(self, index):
        from dataset import PromptDataset
        return PromptDataset.__getitem__(self, index)


def compute_conditions(va, ta, pqa, align):
    """Raw [B,H,W] -> three raw [B,H,W] maps; use the original fusion_fun."""
    from adaptcliplib import fusion_fun
    average = fusion_fun([va, ta, pqa], fusion_type="average_mean")
    return {
        "baseline": fusion_fun([average, align], fusion_type="harmonic_mean"),
        "no_align": average,
        "pqa_align": fusion_fun([pqa, align], fusion_type="harmonic_mean"),
    }


def map_statistics(shot, sample, name, stage, values, mask):
    """Summarize one unnormalized [H,W] map; empty defect regions are N/A."""
    import numpy as np
    flat = values.ravel()
    fg, bg = values[mask > 0], values[mask == 0]
    mean = lambda array: float(array.mean(dtype=np.float64)) if array.size else None
    percentile = lambda array: float(np.percentile(array, 95)) if array.size else None
    row = {"Shot": shot, "Category": sample["Category"], "Sample Type": sample["Sample Type"],
           "Image Path": sample["Image Path"], "Map": name, "Stage": stage,
           "Dtype": str(values.dtype), "Height": values.shape[0], "Width": values.shape[1],
           "Min": float(flat.min()), "Max": float(flat.max()), "Mean": mean(flat),
           "Std": float(flat.std(dtype=np.float64)), "P50": float(np.percentile(flat, 50)),
           "P95": percentile(flat), "Foreground Mean": mean(fg),
           "Foreground Max": float(fg.max()) if fg.size else None, "Foreground P95": percentile(fg),
           "Background Mean": mean(bg), "Background P95": percentile(bg),
           "Foreground Minus Background": mean(fg) - mean(bg) if fg.size and bg.size else None,
           "Top 1% Mean": None}
    if sample["Sample Type"] == "Normal":
        count = max(1, math.ceil(flat.size * 0.01))
        row["Top 1% Mean"] = mean(np.partition(flat, flat.size - count)[-count:])
    return row


def save_panels(path, image, mask, raw, evaluated, names, columns, title):
    """Plot aligned model input/GT/maps with a FIXED [0,1] heatmap scale across shots.

    Score arrays are never normalized or clipped before metrics/statistics/saving.
    Titles report raw/evaluated extrema, including values outside display bounds.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adaptcliplib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * np.asarray(OPENAI_DATASET_STD) + np.asarray(OPENAI_DATASET_MEAN), 0, 1)
    count = 2 + len(names)
    fig, axes = plt.subplots(math.ceil(count / columns), columns,
                             figsize=(columns * 3.6, math.ceil(count / columns) * 3.6), squeeze=False)
    axes = axes.ravel()
    try:
        axes[0].imshow(rgb)
        axes[0].set_title("Original (model input)")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        for axis, name in zip(axes[2:], names):
            plotted = axis.imshow(evaluated[name], cmap="turbo", vmin=0, vmax=1)
            axis.set_title("FINAL (baseline)" if name == "baseline" else name)
            axis.set_xlabel(f"raw {raw[name].min():.5g} .. {raw[name].max():.5g}\n"
                            f"eval {evaluated[name].min():.5g} .. {evaluated[name].max():.5g}", fontsize=8)
        for index, axis in enumerate(axes):
            axis.set_xticks([])
            axis.set_yticks([])
            if index >= count:
                axis.set_visible(False)
        fig.suptitle(title)
        fig.subplots_adjust(left=0.02, right=0.91, top=0.91, bottom=0.10, hspace=0.45, wspace=0.12)
        color_axis = fig.add_axes([0.94, 0.18, 0.012, 0.62])
        fig.colorbar(plotted, cax=color_axis, label="Smoothed score; fixed display range [0,1]")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)


def metric_tables(output, pixels, images):
    """Save full precision category rows plus equal-weight category means."""
    from branch_diagnostics_utils import with_macro_means
    all_pixels = with_macro_means(pixels, ["Condition"])
    all_images = with_macro_means(images, ["Condition"])
    for name, rows in (("pixel_metrics.csv", all_pixels), ("image_metrics.csv", all_images)):
        write_csv(output / name, ["Category", "Condition", *METRICS], rows)
    combined = {}
    for level, rows in (("P", all_pixels), ("I", all_images)):
        for row in rows:
            key = (row["Category"], row["Condition"])
            combined.setdefault(key, {"Category": key[0], "Condition": key[1]})
            combined[key].update({f"{level}-{m}": row[m] for m in METRICS})
    return list(combined.values())


def compare_baseline(source, combined, tolerance):
    """Compare every category and macro metric with full-precision stored FINAL."""
    checks = []
    for row in combined:
        if row["Condition"] != "baseline":
            continue
        for metric in [f"{level}-{name}" for level in ("I", "P") for name in METRICS]:
            actual = row[metric]
            expected = source["metrics"][(row["Category"], metric)]
            delta = actual - expected if actual is not None else None
            checks.append({"Category": row["Category"], "Metric": metric, "Recorded": expected,
                           "Current": actual, "Delta": delta, "Tolerance pp": tolerance,
                           "Passed": delta is not None and abs(delta) <= tolerance})
    return checks
