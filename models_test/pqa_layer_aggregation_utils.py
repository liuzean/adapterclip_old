"""Conservative spatial-agreement PQA aggregation and four-condition evaluation.

Only standard-library dependencies are imported before numerical calls.
"""

import math

from dual_output_utils import FIELDS
from fusion_ablation_utils import METRICS

CONDITIONS = ("baseline", "dual_output", "pqa_weighted", "pqa_weighted_dual")
PIXEL_SOURCES = dict(zip(CONDITIONS, ("baseline", "no_TA", "weighted", "weighted_no_TA")))
IMAGE_SOURCES = dict(zip(CONDITIONS, ("baseline", "baseline", "weighted", "weighted")))
GRID_SIZE = 32


def aggregate_layers(layers, strength=0.5):
    """[B,4,H,W] probabilities -> original mean, weighted map and small metadata.

    Per sample, area-pool each layer to 32x32 in float64, center and normalize
    its spatial pattern. Positive Pearson correlations with other nonconstant
    layers define agreement. Normalize agreement into q, then use
    w=(1-strength)/4 + strength*q. No positive agreement -> uniform fallback.
    Only the weight calculation uses pooled/normalized values; final maps are
    convex combinations of ORIGINAL full-resolution probabilities and dtype.
    At strength=0, return the original mean tensor directly for exact recovery.
    Labels, categories, batch statistics, logits and trainable state are absent.
    """
    import torch
    from torch.nn.functional import adaptive_avg_pool2d
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("aggregation_strength must be finite and in [0,1].")
    if layers.ndim != 4 or layers.shape[1] != 4 or not layers.is_floating_point() or min(layers.shape) < 1:
        raise ValueError("Expected nonempty floating [B,4,H,W] PQA maps.")
    if not torch.isfinite(layers).all():
        raise ValueError("Nonfinite PQA layer maps.")
    original = layers.mean(dim=1)
    uniform = torch.full((layers.shape[0], 4), .25, dtype=torch.float64, device=layers.device)
    if strength == 0:
        return original, original, {"weights": uniform, "agreement": torch.zeros_like(uniform),
                                    "fallback": torch.ones(layers.shape[0], dtype=torch.bool, device=layers.device)}
    # Pool one layer at a time to avoid allocating full [B,4,H,W] float64 maps.
    pooled = torch.stack([adaptive_avg_pool2d(layers[:, i:i+1].detach().double(), (GRID_SIZE, GRID_SIZE))[:, 0]
                          for i in range(4)], dim=1).flatten(2)
    centered = pooled - pooled.mean(dim=-1, keepdim=True)
    scale = centered.abs().amax(dim=-1, keepdim=True)
    valid = scale.squeeze(-1) > 0
    scaled = centered / torch.where(scale > 0, scale, torch.ones_like(scale))
    norm = scaled.square().sum(dim=-1, keepdim=True).sqrt()
    normalized = scaled / torch.where(norm > 0, norm, torch.ones_like(norm))
    correlation = torch.bmm(normalized, normalized.transpose(1, 2)).clamp(-1, 1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    pair_valid &= ~torch.eye(4, device=layers.device, dtype=torch.bool).unsqueeze(0)
    peers = pair_valid.sum(dim=-1)
    agreement = (correlation.clamp_min(0) * pair_valid).sum(dim=-1) / peers.clamp_min(1)
    total = agreement.sum(dim=1, keepdim=True)
    fallback = total.squeeze(1) <= torch.finfo(torch.float64).eps
    q = agreement / torch.where(total > 0, total, torch.ones_like(total))
    q = torch.where(fallback.unsqueeze(1), uniform, q)
    weights = (1 - strength) * uniform + strength * q
    weighted = (layers * weights.to(layers.dtype).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    # Also preserve the original reduction on samples lacking agreement evidence.
    weighted = torch.where(fallback[:, None, None], original, weighted)
    if not torch.isfinite(weighted).all() or not torch.isfinite(weights).all():
        raise ValueError("Nonfinite weighted PQA aggregation.")
    if not torch.allclose(weights.sum(dim=1), torch.ones_like(total[:, 0]), atol=1e-12, rtol=0):
        raise ValueError("Layer weights must sum to one.")
    return original, weighted, {"weights": weights, "agreement": agreement, "fallback": fallback}


def infer_batch(config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths, strength):
    """One original forward per Adapter -> four pixel variants and routed scores.

    Score extraction mirrors branch_ablation.infer_batch/test.py. No model
    forward, matching, BN, head, residual or native-global logic is replaced.
    Returned maps [B,H,W]/scores [B] are CPU; raw maps remain on model device.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_ablation_utils import compute_conditions
    from branch_diagnostics_utils import require_finite, smooth_maps
    vg, vl = visual(features, patches, textual.static_text_features)
    tg, tl = textual.compute_global_local_score(features, patches, learned_text)
    pg, pl, distances = pq(features, patches, prompt_features, prompt_patches)
    if len(pl) != 4 or len(distances) != 4:
        raise ValueError("Expected four original PQA/ALIGN maps.")
    layers = torch.concat([value[:, 1].detach().unsqueeze(1) for value in pl], dim=1)
    original, weighted, details = aggregate_layers(layers, strength)
    align = fusion_fun(distances, fusion_type="harmonic_mean")[:, 0]
    old = compute_conditions(vl[:, 1].detach(), tl[:, 1].detach(), original, align)
    new = compute_conditions(vl[:, 1].detach(), tl[:, 1].detach(), weighted, align)
    raw = {"baseline": old["baseline"], "no_TA": old["no_TA"],
           "weighted": new["baseline"], "weighted_no_TA": new["no_TA"]}
    if isinstance(pg, list):
        pqa_global = torch.concat([value.softmax(-1).unsqueeze(-1) for value in pg], dim=-1).mean(dim=-1).detach()[:, 1]
    else:
        pqa_global = pg.softmax(-1)[:, 1].detach()
    globals_ = {"VA": vg.softmax(-1)[:, 1].detach(), "TA": tg.softmax(-1)[:, 1].detach(), "PQA": pqa_global}
    fixed = fusion_fun(list(globals_.values()), fusion_type="average_mean")
    for name, value in {**globals_, "fixed": fixed, "uniform PQA": original, "weighted PQA": weighted, "ALIGN": align}.items():
        require_finite(name, value, paths)
    maps, maxima, variant_scores = {}, {}, {}
    for name, value in raw.items():
        require_finite(name, value, paths)
        smoothed = smooth_maps(value, config.sigma).to(features.device)
        maximum = smoothed.view(features.shape[0], -1).max(dim=1)[0]
        score = fusion_fun([fixed, maximum], fusion_type="harmonic_mean")
        require_finite(f"{name} smoothed", smoothed, paths)
        require_finite(f"{name} score", score, paths)
        maps[name] = torch.nan_to_num(smoothed, nan=0., posinf=0., neginf=0.).cpu()
        variant_scores[name] = torch.nan_to_num(score, nan=0., posinf=0., neginf=0.).cpu()
        maxima[name] = maximum.cpu()
    if strength == 0 and any(not torch.equal(maps[a], maps[b]) or not torch.equal(variant_scores[a], variant_scores[b])
                             for a, b in (("baseline", "weighted"), ("no_TA", "weighted_no_TA"))):
        raise ValueError("Disabled aggregation failed exact output recovery.")
    return (raw, maps, variant_scores, maxima, globals_, fixed,
            {key: value.detach().cpu() for key, value in details.items()})


def compare_controls(recorded, combined, tolerance):
    """Original baseline/no_TA records determine expected two control conditions."""
    checks = []
    for row in combined:
        if row["Condition"] not in ("baseline", "dual_output"):
            continue
        for metric in FIELDS:
            source = "no_TA" if row["Condition"] == "dual_output" and metric.startswith("P-") else "baseline"
            expected = float(recorded["metrics"][(row["Category"], source)][metric])
            value = row[metric]
            delta = value - expected if value is not None else None
            checks.append({"Category": row["Category"], "Condition": row["Condition"], "Metric": metric,
                           "Recorded": expected, "Current": value, "Delta": delta,
                           "Passed": delta is not None and abs(delta) <= tolerance})
    return checks


class CategoryBuffer:
    """Four unique pixel maps and two unique score vectors, bounded to one class."""

    def __init__(self, category):
        self.category = category
        self.masks, self.labels = [], []
        self.maps = {name: [] for name in PIXEL_SOURCES.values()}
        self.scores = {name: [] for name in ("baseline", "weighted")}

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.to(torch.uint8).clone())
        self.labels.append(label)
        for name in self.maps:
            self.maps[name].append(maps[name][index].clone())
        for name in self.scores:
            self.scores[name].append(scores[name][index].clone())

    def finish(self, device, logger):
        """Reuse original exact metrics; each unique image score is evaluated once."""
        import torch
        from tools import Evaluator
        from branch_diagnostics_utils import evaluate_scores
        masks, labels = torch.stack(self.masks).int(), torch.tensor(self.labels).int()
        image_metrics = {}
        for name in self.scores:
            scores = torch.stack(self.scores[name])
            self.scores[name].clear()
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f"I-{metric}" for metric in METRICS])
            image_metrics[name] = evaluate_scores(evaluator, self.category, dummy, dummy, labels, scores, device, "image")
            del evaluator, dummy, scores
        pixels, images = [], []
        for condition in CONDITIONS:
            name = PIXEL_SOURCES[condition]
            maps = torch.stack(self.maps[name])
            self.maps[name].clear()
            evaluator = Evaluator(device, metrics=[f"P-{metric}" for metric in METRICS])
            dummy_scores = torch.zeros(len(labels))
            pixels.append({"Category": self.category, "Condition": condition, **evaluate_scores(
                evaluator, self.category, masks, maps, labels, dummy_scores, device, "pixel")})
            images.append({"Category": self.category, "Condition": condition, **image_metrics[IMAGE_SOURCES[condition]]})
            del evaluator, maps, dummy_scores
        logger.info("%s: pixel=%s image=%s", self.category, pixels, images)
        self.masks.clear()
        self.labels.clear()
        return pixels, images
