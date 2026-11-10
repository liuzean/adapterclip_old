"""CRA candidate adapted from Reference/TGSA/module.py:ContextResidualAgg.

Only the zero-gated 3x3 depthwise residual design is used. No TGA, SPL,
prompt changes or TGSA checkpoint/decoder is imported.
"""
from contextlib import contextmanager

import torch
from torch import nn


class ContextResidualAggregation(nn.Module):
    """Four per-layer residual operators on PQA [B,768,H,W] compare features."""

    def __init__(self, dim=768, layers=4, kernel=3):
        super().__init__()
        if min(dim, layers) < 1 or kernel != 3:
            raise ValueError('Positive dimensions and the fixed 3x3 kernel required.')
        self.spec = dict(dim=dim, layers=layers, kernel=kernel)
        self.convs = nn.ModuleList([nn.Conv2d(dim, dim, kernel, padding=1, groups=dim, bias=False)
                                    for _ in range(layers)])
        self.gates = nn.Parameter(torch.zeros(layers))

    def forward(self, features, layer):
        if features.ndim != 4 or features.shape[1] != self.spec['dim'] or not 0 <= layer < self.spec['layers']:
            raise ValueError('CRA requires a valid layer and [B,D,H,W] compare features.')
        return features + self.gates[layer].tanh() * self.convs[layer](features)


class PQAResidualHook:
    """Insert CRA immediately BEFORE each original PQA sharebn, without editing it.

    CRA is kept outside PQA's module tree so frozen adapter eval/BN state stays
    immutable while CRA alone is in train mode. Inputs are replaced, never edited.
    Always remove handles when the experiment model is released.
    """

    def __init__(self, pq, cra):
        if (not pq.context or len(pq.sharebn) != cra.spec['layers']
                or any(bn.num_features != cra.spec['dim'] for bn in pq.sharebn)):
            raise ValueError('CRA requires original context=True PQA layer BN interfaces.')
        self.cra, self.enabled, self.calls = cra, True, 0
        self.handles = []
        for layer, bn in enumerate(pq.sharebn):
            def before(module, args, layer=layer):
                if not self.enabled:
                    return None
                if len(args) != 1:
                    raise ValueError('Unexpected PQA normalization input interface.')
                self.calls += 1
                return (self.cra(args[0], layer),)
            self.handles.append(bn.register_forward_pre_hook(before))

    @contextmanager
    def active(self, enabled):
        previous = self.enabled
        self.enabled = bool(enabled)
        try:
            yield
        finally:
            self.enabled = previous

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
