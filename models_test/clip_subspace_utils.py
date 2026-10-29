"""Normal-reference CLIP PCA residuals; no model, sampler or evaluator replacement.

Inspired by SubspaceAD (https://github.com/CLendering/SubspaceAD), with explicit
CLIP feature/normalization/scoring differences. Not an official reproduction.
Only standard-library modules are imported until numerical functions are called.
"""

import math

from dual_output_utils import CategoryBuffer as ControlBuffer
from dual_output_utils import FIELDS, audit_ablation
from fusion_ablation_utils import METRICS, path_key, read_csv, read_json

CONDITIONS = ("baseline", "dual_output", "clip_pca")


def audit_controls(source, directory):
    """Require an actual enabled dual-output run, including its routing metrics."""
    from pathlib import Path
    recorded = audit_ablation(source, directory)
    config = read_json(Path(directory) / "experiment_config.json")
    if config.get("disable_dual_output") is not False:
        raise ValueError("Control must be a completed enabled dual_output run.")
    dual_rows = [row for row in read_csv(Path(directory) / "metrics_combined.csv")
                 if row["Condition"] == "dual_output"]
    expected = {*source["classes"], "Mean"}
    if len(dual_rows) != len(expected) or {row["Category"] for row in dual_rows} != expected:
        raise ValueError("Control dual-output metrics are missing/duplicated.")
    for row in dual_rows:
        for field in FIELDS:
            key = "no_TA" if field.startswith("P-") else "baseline"
            if float(row[field]) != float(recorded["metrics"][(row["Category"], key)][field]):
                raise ValueError("Control dual-output metrics do not match recorded routing.")
    return recorded


def check_reference_separation(source):
    """After audit_source checks normal TRAIN membership, reject test overlap."""
    query_paths = {path_key(row["Image Path"]) for row in source["manifest"]}
    overlap = [path for _, path in source["reference_paths"] if path_key(path) in query_paths]
    if overlap:
        raise ValueError(f"PCA references overlap test paths: {overlap}")


def patch_descriptors(layers):
    """Four [B,1+L,D] -> [B,L,D] unit descriptors; source tensors are read only.

    Remove CLS, take a fixed equal feature mean, then L2-normalize. This is not
    the earlier probability-map agreement weighting. Same operation for support
    and queries; no batch statistics, labels, augmentation or learned weights.
    """
    import torch
    if len(layers) != 4:
        raise ValueError("Expected four CLIP feature layers.")
    shape = layers[0].shape
    if len(shape) != 3 or min(shape) < 1 or shape[1] <= 1:
        raise ValueError("Expected nonempty [B,1+L,D] features.")
    if any(value.shape != shape or value.device != layers[0].device or not value.is_floating_point()
           for value in layers):
        raise ValueError("Feature layers must share shape/device and be floating point.")
    if math.isqrt(shape[1] - 1) ** 2 != shape[1] - 1:
        raise ValueError("Patch tokens must form a square grid after CLS removal.")
    if any(not torch.isfinite(value).all() for value in layers):
        raise ValueError("Nonfinite CLIP feature layer.")
    value = torch.stack([layer[:, 1:].detach().float() for layer in layers]).mean(dim=0)
    norms = value.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("Cannot normalize nonfinite or zero CLIP patch descriptors.")
    return value / norms


def fit_subspace(references, explained_variance=.99):
    """Fit ONLY [N,D] normal support descriptors; return CPU float64 PCA state.

    One fixed matrix supplies mean and covariance. Numerical zero eigenvalues
    are excluded. Constant support uses a zero-dimensional subspace (distance
    to its mean), explicitly recorded rather than inventing a PCA direction.
    """
    import torch
    if not math.isfinite(explained_variance) or not 0 < explained_variance < 1:
        raise ValueError("pca_ev must be finite and strictly between 0 and 1.")
    if references.ndim != 2 or references.shape[0] < 2 or references.shape[1] < 1:
        raise ValueError("PCA needs [N,D] support with N >= 2.")
    values = references.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite PCA support.")
    if not torch.allclose(values.norm(dim=1), torch.ones(values.shape[0], dtype=values.dtype),
                          atol=1e-5, rtol=0):
        raise ValueError("PCA support descriptors must be unit length.")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues, vectors = eigenvalues.flip(0).clamp_min(0), vectors.flip(1)
    tolerance = float(eigenvalues[0]) * max(values.shape) * torch.finfo(values.dtype).eps
    rank = int((eigenvalues > tolerance).sum())
    total = eigenvalues[:rank].sum()
    if rank:
        cumulative = eigenvalues[:rank].cumsum(dim=0) / total
        count = min(int(torch.searchsorted(cumulative, cumulative.new_tensor(explained_variance))) + 1, rank)
        retained = float(cumulative[count - 1])
    else:
        count, retained = 0, 0.
    basis = vectors[:, :count].contiguous()
    if not torch.isfinite(basis).all() or not torch.isfinite(mean).all():
        raise ValueError("Nonfinite PCA fit.")
    return {"mean": mean, "basis": basis, "eigenvalues": eigenvalues[:rank],
            "metadata": {"tokens": len(values), "dimension": values.shape[1], "rank": rank,
                         "components": count, "target_ev": explained_variance, "retained_ev": retained,
                         "total_variance": float(total), "degenerate": rank == 0}}


def residual_scores(descriptors, state):
    """[N,D] unit queries -> [N] squared orthogonal residual / 4, on query device.

    Unit x and mean(norm<=1) imply ||(I-UU^T)(x-mu)||² <= 4. Fixed scaling keeps
    the evaluator input bounded, without test-set or per-image calibration.
    Explicit residual subtraction avoids subtracting nearly equal squared norms.
    """
    import torch
    if descriptors.ndim != 2 or descriptors.shape[1] != state["mean"].numel():
        raise ValueError("Query/PCA feature dimensions differ.")
    if not torch.isfinite(descriptors).all():
        raise ValueError("Nonfinite query descriptors.")
    if not torch.allclose(descriptors.norm(dim=1), torch.ones_like(descriptors[:, 0]), atol=1e-5, rtol=0):
        raise ValueError("Query descriptors must be unit length.")
    mean = state["mean"].to(device=descriptors.device, dtype=descriptors.dtype)
    basis = state["basis"].to(device=descriptors.device, dtype=descriptors.dtype)
    centered = descriptors - mean
    residual = centered - (centered @ basis) @ basis.T
    scores = residual.square().sum(dim=-1) / 4.
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1 + 1e-5).any():
        raise ValueError("PCA residual violated finite [0,1] bound.")
    return scores.clamp(max=1)  # only numerical overshoot at the theoretical bound


def candidate_outputs(patches, categories, states, image_size, sigma, baseline_raw,
                      baseline_map, baseline_score, disable=False):
    """Return raw/smoothed [B,H,W], [B] image max; disabled outputs alias baseline.

    No fusion with original branches. States were fitted before reading queries.
    Batch boundaries may cross categories: route each query to its own support.
    """
    if disable:
        return baseline_raw, baseline_map, baseline_score
    import torch
    from torch.nn.functional import interpolate
    from branch_diagnostics_utils import smooth_maps
    descriptors = patch_descriptors(patches)
    if len(categories) != descriptors.shape[0]:
        raise ValueError("Query categories/batch size differ.")
    grid = math.isqrt(descriptors.shape[1])
    rows = []
    for index, category in enumerate(categories):
        rows.append(residual_scores(descriptors[index], states[category]).reshape(grid, grid))
    raw = interpolate(torch.stack(rows)[:, None], size=(image_size, image_size),
                      mode="bilinear", align_corners=False)[:, 0]
    evaluated = smooth_maps(raw, sigma)
    scores = evaluated.flatten(1).max(dim=1).values
    if not torch.isfinite(evaluated).all() or not torch.isfinite(scores).all():
        raise ValueError("Nonfinite postprocessed PCA output.")
    return raw, evaluated, scores


def compare_controls(recorded, combined, tolerance):
    """Compare baseline and dual routing with completed historical control CSVs."""
    rows = []
    for row in combined:
        if row["Condition"] not in ("baseline", "dual_output"):
            continue
        for field in FIELDS:
            source = "no_TA" if row["Condition"] == "dual_output" and field.startswith("P-") else "baseline"
            old = float(recorded["metrics"][(row["Category"], source)][field])
            delta = row[field] - old if row[field] is not None else None
            rows.append({"Category": row["Category"], "Condition": row["Condition"], "Metric": field,
                         "Recorded": old, "Current": row[field], "Delta": delta,
                         "Passed": delta is not None and abs(delta) <= tolerance})
    return rows


class CategoryBuffer:
    """Reuse old control evaluator; add only one independent map per current class."""

    def __init__(self, category, disable=False):
        self.category, self.disable = category, disable
        self.control = ControlBuffer(category)
        self.maps, self.scores = [], []

    def append(self, mask, label, maps, scores, index):
        self.control.append(mask, label, maps, scores, index)
        if not self.disable:
            self.maps.append(maps["clip_pca"][index].clone())
            self.scores.append(scores["clip_pca"][index].clone())

    def finish(self, device, logger):
        """Same six metrics; GT only enters here, after predictions are fixed."""
        import torch
        from tools import Evaluator
        from branch_diagnostics_utils import evaluate_scores
        if not self.disable:
            # Hold references to the uint8 masks, not another dense int32 stack
            # during control evaluation. Build the candidate stack afterwards.
            mask_items = tuple(self.control.masks)
            label_values = tuple(self.control.labels)
        pixels, images = self.control.finish(device, logger)
        pixels = [row for row in pixels if row["Condition"] != "no_TA"]
        images = [row for row in images if row["Condition"] != "no_TA"]
        if self.disable:
            pixels.append({**pixels[0], "Condition": "clip_pca"})
            images.append({**images[0], "Condition": "clip_pca"})
        else:
            masks = torch.stack(mask_items).int()
            labels = torch.tensor(label_values).int()
            del mask_items, label_values
            maps, scores = torch.stack(self.maps), torch.stack(self.scores)
            self.maps.clear()
            self.scores.clear()
            evaluator = Evaluator(device, metrics=[f"P-{m}" for m in METRICS])
            pixels.append({"Category": self.category, "Condition": "clip_pca", **evaluate_scores(
                evaluator, self.category, masks, maps, labels, scores, device, "pixel")})
            del evaluator, maps, masks
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f"I-{m}" for m in METRICS])
            images.append({"Category": self.category, "Condition": "clip_pca", **evaluate_scores(
                evaluator, self.category, dummy, dummy, labels, scores, device, "image")})
        logger.info("%s / CLIP PCA: pixel=%s image=%s", self.category, pixels[-1], images[-1])
        return pixels, images
