"""CPU regression tests for experiment M contracts; no weights/datasets needed.

Run explicitly with: python -m pytest tests/test_pqa_local_control_contracts.py
These tests do not evaluate algorithm effectiveness or launch training.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))

from pqa_local_control_utils import FIELDS, select_source
from prepare_paper_evaluation import complete_dataset_means, validate_external


def test_selection_uses_only_pixel_metrics_and_requires_three_shots():
    rows = [dict(Category='Mean', Condition='adapted', Shot=s,
                 **{f:(90 if f.startswith('P-') else 1) for f in FIELDS}) for s in (1,2,4)]
    assert select_source(rows) == 90
    for row in rows:
        for f in FIELDS[:3]: row[f] = 99
    assert select_source(rows) == 90
    with pytest.raises(ValueError): select_source(rows[:-1])


def test_external_partial_or_duplicate_metrics_are_not_accepted():
    row = dict(Method='SubspaceAD', Dataset='Visa', Shot=1, Seed=10, **{f:80 for f in FIELDS})
    assert validate_external([row])[0]['ProtocolStatus'] == 'unverified_import'
    with pytest.raises(ValueError): validate_external([row,row])
    row[FIELDS[0]] = ''
    with pytest.raises(ValueError): validate_external([row])


def test_external_partial_dataset_set_has_no_six_dataset_mean():
    row = dict(Method='SubspaceAD', Dataset='Visa', Shot=1, Seed=10, **{f:80 for f in FIELDS})
    assert complete_dataset_means([row]) == []


def test_shared_head_copy_has_130_parameters_and_is_independent():
    torch = pytest.importorskip('torch')
    from pqa_local_control_model import FeatureTap, classify, make_classifier
    seq = torch.nn.Sequential(torch.nn.BatchNorm2d(64), torch.nn.Conv2d(64,2,1))
    pq = torch.nn.Module(); pq.local_adapter = torch.nn.ModuleList([seq]*4)
    pq.eval().requires_grad_(False)
    head = make_classifier(pq)
    assert sum(p.numel() for p in head.parameters()) == 130
    original = {k:v.clone() for k,v in pq.state_dict().items()}
    tap = FeatureTap(pq)
    try:
        with torch.no_grad(), tap.capture() as captured:
            old = [pq.local_adapter[i](torch.randn(2,64,3,3)).softmax(1) for i in range(4)]
        from torch.nn.functional import interpolate
        expected = torch.cat([interpolate(v,(9,9),mode='bilinear')[:,1,None] for v in old],dim=1).mean(1)
        predicted = classify(head, captured, 9)
        assert torch.equal(predicted,expected)
        predicted.mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        with torch.no_grad(): next(head.parameters()).add_(.1)
        assert all(torch.equal(value,original[name]) for name,value in pq.state_dict().items())
        assert all(p.grad is None for p in pq.parameters())
        assert not any(m.training for m in pq.modules())
    finally:
        tap.close()


def test_local_fusion_gradient_reaches_only_new_classifier(monkeypatch):
    torch = pytest.importorskip('torch')
    from pqa_local_control_model import classify, local_raw
    # Isolate gradient connectivity from heavyweight CLIP imports in this CPU test.
    def fusion(values, fusion_type):
        stacked = torch.stack(values)
        return stacked.mean(0) if fusion_type == 'average_mean' else len(values)/(1/(stacked+1e-8)).sum(0)
    monkeypatch.setitem(sys.modules,'adaptcliplib',SimpleNamespace(fusion_fun=fusion))
    head = torch.nn.Conv2d(64,2,1)
    frozen = [torch.randn(2,64,3,3) for _ in range(4)]
    base = torch.full((2,9,9),.4)
    raw = local_raw(base,base,classify(head,frozen,9),base,base)
    raw.square().mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in head.parameters())
    assert all(x.grad is None for x in frozen)
