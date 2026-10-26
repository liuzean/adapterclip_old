"""Fixed samples and read-only PQA observations for branch_ablation.

No tensor or model dependencies are imported by the parent scheduler.
"""

from collections import Counter, defaultdict

from fusion_ablation_utils import METRICS, STAT_FIELDS, path_key

CONDITIONS = ("baseline", "no_VA", "no_TA")
LAYER_STAT_FIELDS = (*STAT_FIELDS, "Zero Fraction", "Below 1e-12 Fraction", "Global Peak In Defect")
INTERNAL_FIELDS = (
    "Shot", "Category", "Sample Type", "Image Path", "Layer", "Stage", "Dtype",
    "Shape", "Min", "Max", "Mean", "Std", "Abs Mean", "RMS", "Zero Fraction",
    "Negative Fraction", "Below 1e-12 Fraction", "Above 1-1e-6 Fraction",
)


def select_samples(sources, per_type):
    """Select the first N normal/anomalous entries per class in saved test order.

    Selection never uses predictions or metric values. Keep source rows intact,
    including old map paths, and require identical identities across shots.
    """
    if per_type < 1 or not sources:
        raise ValueError("Need at least one source and a positive diagnostic_per_type.")
    first = None
    for source in sources:
        counts = Counter()
        selected = []
        for row in source["manifest"]:
            key = (row["Category"], row["Sample Type"])
            if row["Sample Type"] not in ("Normal", "Anomaly"):
                raise ValueError(f"Unknown sample type: {row['Sample Type']}")
            if counts[key] < per_type:
                selected.append(dict(row))
                counts[key] += 1
        expected = Counter({(category, label): per_type for category in source["classes"]
                            for label in ("Normal", "Anomaly")})
        if counts != expected:
            raise ValueError("Not enough fixed normal/anomalous samples in every class.")
        identities = [(r["Category"], r["Sample Type"], path_key(r["Image Path"]), r["Sample Key"])
                      for r in selected]
        if len({row[2] for row in identities}) != len(identities):
            raise ValueError("Duplicate diagnostic image identity.")
        if first is not None and identities != first:
            raise ValueError("Diagnostic identities/order differ across shots.")
        first = identities
        source["diagnostics"] = selected
    return {"policy": "First N normal and N anomalous rows per class in source manifest; no score selection",
            "per_class_per_type": per_type, "sample_count_per_shot": len(first),
            "identical_across_requested_shots": True,
            "samples": [{key: row[key] for key in ("Category", "Sample Type", "Image Path", "Sample Key")}
                        for row in sources[0]["diagnostics"]]}


def compute_conditions(va, ta, pqa, align):
    """Four [B,H,W] maps -> three pixel ablations using original fusion_fun.

    Removing one local branch renormalizes the remaining mean from /3 to /2.
    ALIGN and the original native-global policy are retained in all conditions.
    """
    from adaptcliplib import fusion_fun
    inputs = {"baseline": [va, ta, pqa], "no_VA": [ta, pqa], "no_TA": [va, pqa]}
    return {name: fusion_fun([fusion_fun(values, fusion_type="average_mean"), align],
                             fusion_type="harmonic_mean") for name, values in inputs.items()}


def tensor_statistics(value):
    """Reduce one selected tensor without modifying it or retaining its storage.

    Double precision is used only for diagnostic reductions. The original model
    continues with its original tensor and dtype. Returned fields are scalars.
    """
    import torch
    if not value.numel() or not torch.isfinite(value).all():
        raise ValueError("Empty or nonfinite tensor in PQA diagnostic hook.")
    work = value.detach().to(dtype=torch.float64)
    return {"Dtype": str(value.dtype), "Shape": str(list(value.shape)),
            "Min": float(work.min()), "Max": float(work.max()), "Mean": float(work.mean()),
            "Std": float(work.std(unbiased=False)), "Abs Mean": float(work.abs().mean()),
            "RMS": float(work.square().mean().sqrt()), "Zero Fraction": float((work == 0).double().mean()),
            "Negative Fraction": float((work < 0).double().mean())}


class PQALayerObserver:
    """Observe the REAL PQAdapter forward with temporary non-mutating hooks.

    sharebn[i] supplies the active feature-layer index. Register a local-head
    hook once per unique module because the stable code shares that module at
    all four positions. Each forward must visit BN/head pairs in layer order.
    No Top-1 search or Adapter forward is reimplemented or repeated.
    """

    def __init__(self, pq, features_list, shot, selected_indices):
        self.pq, self.layers, self.shot = pq, tuple(features_list), shot
        self.selected = dict(selected_indices)
        self.handles, self.rows = [], []
        self.bn_seen, self.head_seen = [], []
        self.active = None

    def _record(self, sample, layer, stage, value):
        row = {"Shot": self.shot, "Category": sample["Category"], "Sample Type": sample["Sample Type"],
               "Image Path": sample["Image Path"], "Layer": layer, "Stage": stage,
               **tensor_statistics(value), "Below 1e-12 Fraction": None, "Above 1-1e-6 Fraction": None}
        if stage == "local_probability_before_resize":
            row["Below 1e-12 Fraction"] = float((value < 1e-12).double().mean())
            row["Above 1-1e-6 Fraction"] = float((value > 1 - 1e-6).double().mean())
        self.rows.append(row)

    def _bn_hook(self, index):
        def observe(module, inputs, output):
            if index != len(self.bn_seen) or len(self.head_seen) != index:
                raise ValueError("Unexpected PQA BN/head call order; cannot attribute layers safely.")
            before = inputs[0]
            if before.ndim != 4 or before.shape != output.shape:
                raise ValueError("Expected unchanged [B,D,h,w] shape across PQA sharebn.")
            self.active = index
            self.bn_seen.append(index)
            for item, sample in self.selected.items():
                self._record(sample, self.layers[index], "pq_context_before_bn", before[item])
                self._record(sample, self.layers[index], "pq_context_after_bn", output[item])
            # Returning None preserves the actual model output.
        return observe

    def _head_hook(self, module, inputs, output):
        index = self.active
        if index is None or index != len(self.head_seen) or module is not self.pq.local_adapter[index]:
            raise ValueError("Cannot associate PQA local head output with an observed feature layer.")
        if output.ndim != 4 or output.shape[1] != 2:
            raise ValueError("Expected local logits [B,2,4h,4w] before softmax/interpolation.")
        self.head_seen.append(index)
        for item, sample in self.selected.items():
            logits = output[item].detach()
            self._record(sample, self.layers[index], "local_logit_difference", logits[1] - logits[0])
            # Extra softmax is diagnostic-only on selected outputs, not a second
            # head/model call. Never feed this result back into inference.
            self._record(sample, self.layers[index], "local_probability_before_resize", logits.softmax(dim=0)[1])

    def __enter__(self):
        if not self.selected:
            return self
        if self.pq.training:
            raise ValueError("PQA observations require eval mode; training state must remain unchanged.")
        if len(self.pq.sharebn) != len(self.layers) or len(self.pq.local_adapter) != len(self.layers):
            raise ValueError("PQA observer expects one BN/head position per feature layer.")
        if len({id(module) for module in self.pq.sharebn}) != len(self.layers):
            raise ValueError("Cannot attribute shared BN instances to feature layers.")
        try:
            for index, module in enumerate(self.pq.sharebn):
                self.handles.append(module.register_forward_hook(self._bn_hook(index)))
            seen = set()
            for module in self.pq.local_adapter:
                if id(module) not in seen:
                    self.handles.append(module.register_forward_hook(self._head_hook))
                    seen.add(id(module))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, error_type, error, traceback):
        self._remove_hooks()
        if error_type is None and self.selected:
            expected = list(range(len(self.layers)))
            if self.bn_seen != expected or self.head_seen != expected:
                raise ValueError("Incomplete PQA observation; not all four real layer calls were captured.")
        return False


def map_statistics(shot, sample, name, stage, values, mask):
    """Extend existing unnormalized [H,W] statistics with zero/peak diagnostics."""
    import numpy as np
    from fusion_ablation_utils import map_statistics as original_statistics
    row = original_statistics(shot, sample, name, stage, values, mask)
    peak = np.unravel_index(np.argmax(values), values.shape)
    row.update({"Zero Fraction": float(np.mean(values == 0)),
                "Below 1e-12 Fraction": float(np.mean(values < 1e-12)),
                "Global Peak In Defect": bool(mask[peak] > 0) if np.any(mask > 0) else None})
    return row


def layer_summary(rows):
    """Describe fixed diagnostic images per class/type/layer, not dataset metrics."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in ("Shot", "Category", "Sample Type", "Map", "Stage"))].append(row)
    output = []
    for key, group in groups.items():
        def mean(field):
            values = [row[field] for row in group if row[field] is not None]
            return sum(values) / len(values) if values else None
        output.append({**dict(zip(("Shot", "Category", "Sample Type", "Map", "Stage"), key)),
                       "Sample Count": len(group), "All Zero Sample Count": sum(row["Zero Fraction"] == 1 for row in group),
                       "Max Below 1e-12 Sample Count": sum(row["Max"] < 1e-12 for row in group),
                       "Mean Map Max": mean("Max"), "Mean Map Mean": mean("Mean"),
                       "Mean Foreground Mean": mean("Foreground Mean"), "Mean Background Mean": mean("Background Mean")})
    return output


class CategoryBuffer:
    """Keep one class of three smoothed [H,W] conditions and uint8 GT on CPU."""

    def __init__(self, category):
        self.category = category
        self.masks, self.labels = [], []
        self.maps = {name: [] for name in CONDITIONS}
        self.scores = {name: [] for name in CONDITIONS}

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.to(torch.uint8).clone())
        self.labels.append(label)
        for name in CONDITIONS:
            self.maps[name].append(maps[name][index].clone())
            self.scores[name].append(scores[name][index].clone())

    def finish(self, device, logger):
        """Use the unchanged Evaluator and existing single-label handling."""
        import torch
        from tools import Evaluator
        from branch_diagnostics_utils import evaluate_scores
        masks = torch.stack(self.masks).int()
        labels = torch.tensor(self.labels).int()
        pixels, images = [], []
        for name in CONDITIONS:
            maps = torch.stack(self.maps[name])
            self.maps[name].clear()
            scores = torch.stack(self.scores[name])
            evaluator = Evaluator(device, metrics=[f"P-{metric}" for metric in METRICS])
            pixels.append({"Category": self.category, "Condition": name, **evaluate_scores(
                evaluator, self.category, masks, maps, labels, scores, device, "pixel")})
            del maps, evaluator
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f"I-{metric}" for metric in METRICS])
            images.append({"Category": self.category, "Condition": name, **evaluate_scores(
                evaluator, self.category, dummy, dummy, labels, scores, device, "image")})
            del dummy, evaluator, scores
            self.scores[name].clear()
            logger.info("%s / %s: pixel=%s image=%s", self.category, name, pixels[-1], images[-1])
        self.masks.clear()
        self.labels.clear()
        return pixels, images
