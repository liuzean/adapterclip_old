"""Joint-training contracts; only run on explicit user instruction."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_joint_utils import (
    FullBatchSampler, clone_adapters, joint_forward, joint_losses, require_joint_gradients,
    validate_joint_checkpoint, verify_joint_optimizer,
)
from rma_model import ReferenceMatchingAdapter
from rma_utils import check_frozen, freeze_base


@pytest.mark.parametrize('size,batch', [(7, 2), (9, 4), (2, 8), (8, 2)])
def test_training_batches_keep_every_query_without_singletons(size, batch):
    sampler = FullBatchSampler(list(range(size)), batch)
    groups = list(sampler)
    flattened = [i for group in groups for i in group]
    assert sorted(flattened) == list(range(size))
    assert len(flattened) == len(set(flattened))
    assert all(len(group) >= 2 for group in groups)
    assert len(groups) == len(sampler)


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def encode_text_learn(self, prompts, tokens):
        return self.projection(prompts)


class TinyTextual(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Parameter(torch.randn(2, 4))
        self.register_buffer('static_text_features', torch.ones(2, 4))

    def forward(self):
        return self.context, torch.arange(2)

    def compute_global_local_score(self, features, patches, text):
        logits = features @ text.T
        return logits, logits.softmax(1)[:, :, None, None].expand(-1, -1, 4, 4)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, features, patches, text):
        logits = self.linear(features)
        return logits, logits.softmax(1)[:, :, None, None].expand(-1, -1, 4, 4)


class TinyPQA(TinyVisual):
    def forward(self, features, patches, pf, pp):
        logits, probabilities = super().forward(features, patches, None)
        align = torch.full((len(features), 1, 4, 4), .3)
        align[:, :, 0, 0] = 0.  # Zero ALIGN must not produce NaN training gradients.
        return [logits]*4, [probabilities]*4, [align]*4


def test_all_four_adapters_receive_gradients_while_original_A_stays_unchanged():
    torch.manual_seed(10)
    model, ta, va, pq = TinyTextEncoder(), TinyTextual(), TinyVisual(), TinyPQA()
    initialized = [None, model, ta, va, pq]
    snapshot = freeze_base(initialized)
    original = {k: v.detach().clone() for k, v in ta.state_dict().items()}
    adapters = clone_adapters(initialized, trainable=True)
    rma = ReferenceMatchingAdapter(input_dim=4, hidden_dim=8, image_size=4, chunk=2)
    parameters = [p for module in (*adapters, rma) for p in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    verify_joint_optimizer(snapshot, adapters, rma, optimizer)
    features = torch.randn(2, 4)
    patches = [torch.randn(2, 5, 4) for _ in range(4)]
    pp = [torch.randn(2, 2, 5, 4) for _ in range(4)]
    outputs = joint_forward(model, adapters, rma, features, patches, torch.randn(2, 2, 4), pp)
    focal = lambda p, gt: (p[:, 1]-gt).square().mean()
    dice = lambda p, gt: (p-gt).square().mean()
    total, parts = joint_losses(*outputs, torch.tensor([0, 1]), torch.randint(0, 2, (2, 4, 4)).float(), focal, dice, 1.)
    assert torch.isfinite(total) and all(torch.isfinite(v) for v in parts.values())
    total.backward()
    require_joint_gradients(adapters, rma)
    assert adapters[0].context.grad is not None  # Gradient crosses frozen text encoder.
    optimizer.step()
    check_frozen(snapshot, optimizer)
    assert all(torch.equal(v, original[k]) for k, v in ta.state_dict().items())
    assert not torch.equal(adapters[0].context, ta.context)


def test_joint_loader_rejects_B_checkpoints_and_missing_adapter_states():
    with pytest.raises(ValueError, match='own joint checkpoint'):
        validate_joint_checkpoint(dict(format='adaptclip-rma-v1'), '', '', {})
    with pytest.raises(ValueError, match='all four adapters'):
        validate_joint_checkpoint(dict(format='adaptclip-rma-joint-v1', training_scheme='C_joint', rma={}), '', '', {})
