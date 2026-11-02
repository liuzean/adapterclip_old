"""Trainable reference matching branch; no changes to the original adapters."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceMatchingAdapter(nn.Module):
    """Four shared layer heads on frozen projected CLIP patch features.

    Queries: four [B,1+L,768]; supports: four [B,S,1+L,768].
    Returns image logits [B,2] and pixel logits [B,2,H,W]. Top-K indices
    use frozen cosine features; Q/K/V projections and relation heads learn.
    """

    def __init__(self, input_dim=768, hidden_dim=128, top_k=8,
                 image_size=518, chunk=128, matching='learned'):
        super().__init__()
        if hidden_dim < 8 or hidden_dim % 8 or min(top_k, chunk, image_size) < 1:
            raise ValueError('Positive sizes and hidden_dim divisible by eight required.')
        if matching not in ('learned', 'uniform', 'top1'):
            raise ValueError('Unknown matching mode.')
        self.spec = dict(input_dim=input_dim, hidden_dim=hidden_dim, top_k=top_k,
                         image_size=image_size, chunk=chunk, matching=matching)
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.relation = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1), nn.GroupNorm(8, hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GroupNorm(8, hidden_dim), nn.GELU())
        self.local_head = nn.Conv2d(hidden_dim, 2, 1)
        self.global_head = nn.Linear(hidden_dim * 2, 2)

    def layer(self, queries, supports):
        """Chunk retrieval; train weights only within the same Top-K candidate set."""
        b, tokens, width = queries.shape
        if width != self.spec['input_dim'] or supports.shape[0] != b or supports.shape[-2:] != (tokens, width):
            raise ValueError('Query/support feature interface mismatch.')
        q = F.normalize(queries[:, 1:].detach().float(), dim=-1)
        r = F.normalize(supports[:, :, 1:].detach().float().reshape(b, -1, width), dim=-1)
        side = math.isqrt(q.shape[1])
        if side * side != q.shape[1]:
            raise ValueError('Patch tokens must form a square grid.')
        query, key, value = self.query(q), self.key(r), self.value(r)
        matched = []
        k = min(1 if self.spec['matching'] == 'top1' else self.spec['top_k'], r.shape[1])
        batch = torch.arange(b, device=q.device)[:, None, None]
        for start in range(0, q.shape[1], self.spec['chunk']):
            end = min(start + self.spec['chunk'], q.shape[1])
            with torch.no_grad():
                similarity, indices = (q[:, start:end] @ r.transpose(1, 2)).topk(k, dim=-1)
            keys, values = key[batch, indices], value[batch, indices]
            if self.spec['matching'] == 'learned':
                logits = (query[:, start:end, None] * keys).sum(-1) / math.sqrt(key.shape[-1])
                weights = (logits + similarity).softmax(-1)
            else:
                weights = torch.full_like(similarity, 1.0 / k)
            matched.append((weights[..., None] * values).sum(2))
        reference = torch.cat(matched, dim=1)
        relation = torch.cat((query, reference, (query-reference).abs()), dim=-1)
        relation = self.relation(relation.transpose(1, 2).reshape(b, -1, side, side))
        pooled = torch.cat((relation.mean((2, 3)), relation.amax((2, 3))), dim=-1)
        return self.global_head(pooled), self.local_head(relation)

    def forward(self, queries, supports):
        if len(queries) != 4 or len(supports) != 4:
            raise ValueError('Exactly four feature layers are required.')
        pairs = [self.layer(q, r) for q, r in zip(queries, supports)]
        global_logits = torch.stack([g for g, _ in pairs]).mean(0)
        local_logits = torch.stack([p for _, p in pairs]).mean(0)
        local_logits = F.interpolate(local_logits, size=(self.spec['image_size'],) * 2,
                                     mode='bilinear', align_corners=True)
        return global_logits, local_logits
