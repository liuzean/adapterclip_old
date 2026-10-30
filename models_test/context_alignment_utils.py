"""Read-only feature taps and factorial ALIGN variants; no top-level tensor imports."""

import math

CONDITIONS = ('baseline', 'B_768_context', 'C_1024_point', 'D_1024_context')
PAIRS = (('B_768_context', 'baseline'), ('C_1024_point', 'baseline'),
         ('D_1024_context', 'baseline'), ('D_1024_context', 'C_1024_point'),
         ('D_1024_context', 'B_768_context'))


class RawTap:
    """Observe exactly the four pre-ln_post/proj tensors, without replacing output.

    Transformer outputs ([x,x_ori], list of [1+L,B,1024]). Each activation is
    cloned because later projection must not alias a mutable hidden feature.
    Support mode moves copies to CPU; query mode retains only the current batch.
    """
    def __init__(self, transformer, cpu=False):
        self.batches, self.cpu = [], cpu
        self.handle = transformer.register_forward_hook(self.observe)

    def observe(self, module, inputs, output):
        if len(output[1]) != 4:
            raise ValueError('Expected four transformer feature layers.')
        values = [v.detach().permute(1, 0, 2).contiguous().clone() for v in output[1]]
        if any(v.ndim != 3 or v.shape[-1] != 1024 for v in values):
            raise ValueError('Raw interface must have width 1024.')
        self.batches.append([v.cpu() for v in values] if self.cpu else values)
        # No return: PyTorch must keep the original forward output untouched.

    def take(self):
        if len(self.batches) != 1:
            raise ValueError('Expected exactly one encoder call for this query batch.')
        return self.batches.pop()

    def close(self):
        self.handle.remove()


def initialize_both(source, output, workers, logger):
    """Reuse exact support encoder calls/order/batches; no second encoding pass."""
    import torch
    import test
    from unittest.mock import patch
    from fusion_ablation import initialize
    original = test.build_prompt_memory
    raw_memory, call_count = {}, []

    def build(model, *args, **kwargs):
        tap = RawTap(model.visual.transformer, cpu=True)
        try:
            memory = original(model, *args, **kwargs)
            call_count.append(len(tap.batches))
            layers = [torch.cat([batch[i] for batch in tap.batches], dim=0) for i in range(4)]
            if any(v.shape[0] != len(source['references']) for v in layers):
                raise ValueError('Raw support count/order cannot be matched to recorded references.')
            for category in source['classes']:
                indices = [i for i, (c, _) in enumerate(source['reference_paths']) if c == category]
                raw_memory[category] = [v[indices].clone() for v in layers]
                if any(v.shape[:2] != memory[1][category][i].shape[:2] for i, v in enumerate(raw_memory[category])):
                    raise ValueError('Raw/projected support identity dimensions differ.')
            return memory
        finally:
            tap.close()

    # This temporary experiment-only wrapper observes the original builder.
    # No sampler, encoding batch, model forward, or on-disk source is replaced.
    with patch.object(test, 'build_prompt_memory', build):
        initialized = initialize(source, output, workers, logger)
    if len(call_count) != 1:
        raise ValueError('Expected one original support-memory construction.')
    expected = math.ceil(len(source['references']) / source['config']['batch_size'])
    if call_count[0] != expected:
        raise ValueError('Support encoder call count differs from the original batches.')
    return initialized, raw_memory, call_count[0]


def descriptors(tokens, kernel):
    """[N,1+L,D] -> [N,L,D], pool each image separately then L2 normalize.

    kernel=1 bypasses pooling. kernel=3 uses explicit zero padding with the
    full 3x3 denominator, following DictAS Padding_same; no CLS or cross-image
    mixing. Raw 1024 is BEFORE LayerNorm/projection, projected 768 is AFTER.
    """
    import torch
    from torch.nn import functional as F
    if tokens.ndim != 3 or kernel not in (1, 3):
        raise ValueError('Expected [N,1+L,D] and kernel 1 or 3.')
    grid = math.isqrt(tokens.shape[1]-1)
    if grid*grid != tokens.shape[1]-1 or grid < 1 or not torch.isfinite(tokens).all():
        raise ValueError('Invalid/nonfinite patch grid.')
    value = tokens[:, 1:].detach().float()
    if kernel == 3:
        value = value.transpose(1, 2).reshape(len(value), value.shape[-1], grid, grid)
        value = F.avg_pool2d(F.pad(value, (1, 1, 1, 1)), 3, stride=1)
        value = value.flatten(2).transpose(1, 2)
    norms = value.norm(dim=-1)
    if (norms <= 1e-12).any():
        raise ValueError('Zero/near-zero descriptor; reject instead of inventing a match.')
    return F.normalize(value, dim=-1)


def nearest_distance(query, support, chunk):
    """[L,D], [S*L,D] -> [L] exact nearest cosine distances in query chunks."""
    import torch
    if chunk < 1 or query.ndim != 2 or support.ndim != 2 or query.shape[1] != support.shape[1]:
        raise ValueError('Invalid nearest-neighbor dimensions/chunk.')
    values = [1-(part @ support.T).max(dim=-1).values for part in query.split(chunk)]
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite cosine distance.')
    return result


class ContextMatcher:
    """Cache descriptors only for the currently visited category/layer/interface.

    Supports preserve original k and order. Reference descriptors remain CPU
    cached and are copied per layer/category, bounding GPU matching storage.
    """
    def __init__(self, projected_memory, raw_memory, chunk=256):
        self.projected, self.raw = projected_memory, raw_memory
        self.chunk, self.cache = chunk, {}

    def support(self, category, layer, width, kernel, device):
        cache_key = (category, layer, width, kernel)
        if cache_key not in self.cache:
            tokens = (self.projected if width == 768 else self.raw)[category][layer]
            # Compute both query/support normalization on the same device, then
            # keep only the reusable support descriptors on CPU between batches.
            self.cache[cache_key] = descriptors(tokens.to(device), kernel).flatten(0, 1).cpu()
        return self.cache[cache_key].to(device)

    def alignments(self, patches, raw_patches, categories, original_layers, image_size):
        """Return B/C/D [B,H,W] ALIGN maps; A uses original layers untouched.

        Per layer: B=(old768+context768)/2, C=raw1024 point,
        D=(raw1024 point+context1024)/2. Original harmonic layer aggregation.
        """
        import torch
        from torch.nn import functional as F
        from adaptcliplib import fusion_fun
        if len(patches) != 4 or len(raw_patches) != 4 or len(original_layers) != 4:
            raise ValueError('Four matching layers required.')
        self.cache = {k: v for k, v in self.cache.items() if k[0] in categories}
        layer_maps = {name: [] for name in CONDITIONS[1:]}
        for layer, (projected, raw, old) in enumerate(zip(patches, raw_patches, original_layers)):
            if projected.shape[:2] != raw.shape[:2] or projected.shape[-1] != 768 or raw.shape[-1] != 1024:
                raise ValueError('Unexpected paired interface shapes.')
            grid = math.isqrt(raw.shape[1]-1)
            variants = {}
            for width, kernel, tokens in [(768, 3, projected), (1024, 1, raw), (1024, 3, raw)]:
                query = descriptors(tokens, kernel)
                distances = []
                references = {category: self.support(category, layer, width, kernel, query.device)
                              for category in set(categories)}
                for i, category in enumerate(categories):
                    reference = references[category]
                    distances.append(nearest_distance(query[i], reference, self.chunk).reshape(grid, grid))
                variants[(width, kernel)] = F.interpolate(torch.stack(distances)[:, None],
                    size=(image_size, image_size), mode='bilinear', align_corners=False)
                del references, reference, query
            layer_maps['B_768_context'].append((old + variants[(768, 3)])/2)
            layer_maps['C_1024_point'].append(variants[(1024, 1)])
            layer_maps['D_1024_context'].append((variants[(1024, 1)] + variants[(1024, 3)])/2)
        return {name: fusion_fun(values, fusion_type='harmonic_mean')[:, 0]
                for name, values in layer_maps.items()}


class CategoryBuffer:
    """Four separate outputs, current category only; shared uint8 GT storage."""
    def __init__(self, category):
        self.category, self.masks, self.labels = category, [], []
        self.maps, self.scores = ({name: [] for name in CONDITIONS} for _ in range(2))

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.to(dtype=torch.uint8).clone())
        self.labels.append(label)
        for name in CONDITIONS:
            self.maps[name].append(maps[name][index].clone())
            self.scores[name].append(scores[name][index].clone())

    def finish(self, device, logger):
        import torch
        from tools import Evaluator
        from branch_diagnostics_utils import evaluate_scores
        from fusion_ablation_utils import METRICS
        masks, labels = torch.stack(self.masks).int(), torch.tensor(self.labels).int()
        self.masks.clear()
        self.labels.clear()
        pixels, images = [], []
        for name in CONDITIONS:
            maps, scores = torch.stack(self.maps[name]), torch.stack(self.scores[name])
            self.maps[name].clear()
            self.scores[name].clear()
            evaluator = Evaluator(device, metrics=[f'P-{m}' for m in METRICS])
            pixels.append({'Category': self.category, 'Condition': name, **evaluate_scores(
                evaluator, self.category, masks, maps, labels, scores, device, 'pixel')})
            del evaluator, maps
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f'I-{m}' for m in METRICS])
            images.append({'Category': self.category, 'Condition': name, **evaluate_scores(
                evaluator, self.category, dummy, dummy, labels, scores, device, 'image')})
            logger.info('%s / %s: %s %s', self.category, name, pixels[-1], images[-1])
            del evaluator, dummy, scores
        return pixels, images
