"""Joint-adapter experiment helpers. Parent processes do not import torch."""
import copy
import math

from rma_utils import MetricsBuffer, validation_score, validate_checkpoint_payload

JOINT = 'C_joint'
THREE_ONLY = 'C_joint_without_RMA'
ADAPTER_KEYS = ('textual_learner', 'visual_learner', 'pq_learner')


class FullBatchSampler:
    """Use every source query, merging a trailing singleton for train-mode BN.

    Usually yields batch_size indices; an odd final query can make the last
    batch batch_size+1. No sample is silently dropped or duplicated.
    """

    def __init__(self, sampler, batch_size):
        if batch_size < 2 or len(sampler) < 2:
            raise ValueError('Joint training needs batch_size >= 2 and two source queries.')
        self.sampler, self.batch_size = sampler, batch_size

    def __iter__(self):
        indices = list(self.sampler)
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        n, size = len(self.sampler), self.batch_size
        return math.ceil(n/size) - int(n % size == 1)


def clone_adapters(initialized, trainable=False):
    """Copy ONLY the small three adapters, sharing the single frozen CLIP."""
    adapters = copy.deepcopy(tuple(initialized[2:5]))
    for module in adapters:
        module.requires_grad_(trainable)
        module.train(trainable)
    return adapters


def encode_learned_text(model, textual):
    """Keep the prompt->frozen text encoder path differentiable during training."""
    prompts, tokens = textual()
    return model.encode_text_learn(prompts, tokens).float()


def joint_forward(model, adapters, rma, features, patches, pf, pp):
    """Trainable VA/TA/PQA/RMA outputs from frozen image features.

    Unlike diagnostic infer_batch, NEVER detach adapter probabilities here.
    Only ALIGN is constant because it depends on frozen CLIP matching alone.
    """
    import torch
    from adaptcliplib import fusion_fun
    textual, visual, pq = adapters
    learned_text = encode_learned_text(model, textual)
    va_g, va_p = visual(features, patches, textual.static_text_features)
    ta_g, ta_p = textual.compute_global_local_score(features, patches, learned_text)
    pq_g, pq_p, alignment = pq(features, patches, pf, pp)
    if len(pq_g) != 4 or len(pq_p) != 4 or len(alignment) != 4:
        raise ValueError('Joint training expects the original four PQA layers.')
    rma_g, rma_logits = rma(patches, pp)
    branch_globals = {'VA': [va_g], 'TA': [ta_g], 'PQA': pq_g, 'RMA': [rma_g]}
    branch_pixels = {'VA': [va_p], 'TA': [ta_p], 'PQA': pq_p, 'RMA': [rma_logits.softmax(1)]}
    local_mean = torch.stack([torch.stack(v).mean(0)[:, 1] for v in branch_pixels.values()]).mean(0)
    align = fusion_fun(alignment, fusion_type='harmonic_mean')[:, 0].detach()
    # Algebraic two-input harmonic mean avoids division by zero in backward.
    # It equals 2/(1/a+1/b) for nonzero inputs, and is zero if either is zero.
    numerator = 2 * local_mean * align
    denominator = local_mean + align
    if ((denominator == 0) & (numerator != 0)).any():
        raise ValueError('Singular nonzero harmonic training inputs.')
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    final_raw = numerator / safe_denominator
    return branch_globals, branch_pixels, final_raw


def joint_losses(globals_, pixels, final_raw, labels, masks, focal, dice, fusion_weight):
    """Original VA/TA/PQA loss sums, plus RMA and pre-smoothing fusion loss."""
    import torch
    from torch.nn import functional as F
    components, global_sum, local_sum = {}, 0, 0
    for name in ('VA', 'TA', 'PQA', 'RMA'):
        g = sum(F.cross_entropy(v, labels.long()) for v in globals_[name])
        focal_sum = sum(focal(v, masks) for v in pixels[name])
        anomaly_dice = sum(dice(v[:, 1].contiguous(), masks) for v in pixels[name])
        normal_dice = sum(dice(v[:, 0].contiguous(), 1-masks) for v in pixels[name])
        local = focal_sum + anomaly_dice + normal_dice
        components.update({f'{name}_global_loss': g, f'{name}_local_loss': local,
                           f'{name}_focal_loss': focal_sum, f'{name}_dice_anomaly': anomaly_dice,
                           f'{name}_dice_normal': normal_dice})
        global_sum, local_sum = global_sum+g, local_sum+local
    probabilities = torch.stack((1-final_raw, final_raw), dim=1)
    fusion = focal(probabilities, masks) + dice(final_raw.contiguous(), masks) + dice((1-final_raw).contiguous(), 1-masks)
    total = global_sum + local_sum + fusion_weight*fusion
    return total, dict(global_loss=global_sum, local_loss=local_sum, fusion_loss=fusion, total_loss=total, **components)


def verify_joint_optimizer(snapshot, adapters, rma, optimizer):
    """Exactly the four adapters train; original baseline/CLIP are excluded."""
    from rma_utils import check_frozen
    check_frozen(snapshot, optimizer)
    expected = {id(p) for module in (*adapters, rma) for p in module.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError('Joint optimizer must contain each of the four adapters exactly once.')
    if any(not p.requires_grad for module in (*adapters, rma) for p in module.parameters()):
        raise RuntimeError('A joint adapter remains frozen.')


def require_joint_gradients(adapters, rma):
    """Catch no_grad/detach mistakes in all four participating adapters."""
    import torch
    for name, module in zip(('TA', 'VA', 'PQA', 'RMA'), (*adapters, rma)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
            raise RuntimeError(f'{name} has no finite nonzero training gradient.')


def adapter_states(adapters):
    return {name: module.state_dict() for name, module in zip(ADAPTER_KEYS, adapters)}


def load_joint_adapters(adapters, payload):
    for name, module in zip(ADAPTER_KEYS, adapters):
        module.load_state_dict(payload[name], strict=True)
        module.eval()
        module.requires_grad_(False)


def validate_joint_checkpoint(payload, base_hash, clip_hash, protocol):
    if payload.get('format') != 'adaptclip-rma-joint-v1' or payload.get('training_scheme') != JOINT:
        raise ValueError('C requires its own joint checkpoint, not a B/RMA-only checkpoint.')
    if any(key not in payload for key in (*ADAPTER_KEYS, 'rma')):
        raise ValueError('Joint checkpoint must contain all four adapters.')
    # Reuse shared hash/protocol validation without changing the actual payload.
    validate_checkpoint_payload(dict(payload, format='adaptclip-rma-v1'), base_hash, clip_hash, protocol)


class JointBuffer(MetricsBuffer):
    """Reuse original bounded evaluator while labelling C distinctly from B."""

    def __init__(self, category, candidate=JOINT):
        super().__init__(category)
        self.candidate = candidate

    def append(self, mask, label, maps, scores, index):
        super().append(mask, label, {'baseline': maps['baseline'], 'rma': maps[self.candidate]},
                       {'baseline': scores['baseline'], 'rma': scores[self.candidate]}, index)

    def finish(self, device):
        pixels, images = super().finish(device)
        for row in pixels+images:
            if row['Condition'] == 'rma':
                row['Condition'] = self.candidate
        return pixels, images


def joint_validation_score(tables):
    normalized = {shot: [dict(row, Condition='rma' if row['Condition'] == JOINT else row['Condition'])
                          for row in rows] for shot, rows in tables.items()}
    return validation_score(normalized)


def evaluate_candidate(config, adapters, learned_text, rma, features, patches, pf, pp, paths,
                       without_rma=False):
    """Eval only: updated three adapters + RMA, with the unchanged metric flow."""
    from fusion_ablation import infer_batch
    from rma_utils import fuse_rma
    textual, visual, pq = adapters
    old, maps, scores, _, _, fixed, extra = infer_batch(
        config, features, patches, pf, pp, textual, visual, pq, learned_text, paths)
    if without_rma:
        return old['baseline'], maps['baseline'], scores['baseline']
    g, local = rma(patches, pp)
    return fuse_rma(old, extra, fixed, g, local, config.sigma)
