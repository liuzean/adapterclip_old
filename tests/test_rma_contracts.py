"""Focused RMA contracts. Added for explicit future test runs; not auto-executed."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_model import ReferenceMatchingAdapter
from rma_utils import check_frozen, freeze_base, source_split, validate_checkpoint_payload, validation_score


def test_rma_gradients_leave_input_features_untouched():
    model = ReferenceMatchingAdapter(input_dim=16, hidden_dim=8, top_k=3, image_size=8, chunk=2)
    query = [torch.randn(2, 5, 16, requires_grad=True) for _ in range(4)]
    support = [torch.randn(2, 2, 5, 16, requires_grad=True) for _ in range(4)]
    before = [v.detach().clone() for v in query+support]
    g, p = model(query, support)
    assert g.shape == (2, 2) and p.shape == (2, 2, 8, 8)
    (g.square().mean()+p.square().mean()).backward()
    for tensor, previous in zip(query+support, before):
        assert tensor.grad is None and torch.equal(tensor.detach(), previous)
    for module in (model.query, model.key, model.value, model.relation, model.local_head, model.global_head):
        assert any(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in module.parameters())


def test_learned_matching_changes_when_reference_changes():
    torch.manual_seed(10)
    model = ReferenceMatchingAdapter(input_dim=16, hidden_dim=8, image_size=8, chunk=2).eval()
    query = [torch.randn(1, 5, 16) for _ in range(4)]
    support = [torch.randn(1, 1, 5, 16) for _ in range(4)]
    with torch.no_grad():
        first = model(query, support)
        changed = model(query, [-r for r in support])
    assert any(not torch.allclose(a, b) for a, b in zip(first, changed))


def test_freeze_guard_detects_changed_running_statistics():
    modules = [torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)) for _ in range(4)]
    snapshot = freeze_base([None, *modules])
    check_frozen(snapshot)
    modules[0][1].running_mean.add_(1)
    with pytest.raises(RuntimeError, match='buffers'):
        check_frozen(snapshot)


def test_selection_uses_all_shots_and_full_model_only():
    fields = ('I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max')
    rows = {s: [dict(Category='Mean', Condition='rma', **dict.fromkeys(fields, float(s))),
                dict(Category='Mean', Condition='baseline', **dict.fromkeys(fields, 100.))] for s in (1, 2, 4)}
    assert validation_score(rows) == pytest.approx(7/3)
    invalid = copy.deepcopy(rows)
    invalid[2][0]['I-AP'] = None
    with pytest.raises(ValueError, match='Undefined'):
        validation_score(invalid)


def test_source_split_never_reuses_queries_or_changes_fixed_supports(tmp_path):
    import json
    rows = []
    for label in (0, 1):
        for index in range(5):
            image, mask = f'q_{label}_{index}.png', f'm_{label}_{index}.png'
            (tmp_path/image).touch()
            (tmp_path/mask).touch()
            rows.append(dict(img_path=image, mask_path=mask, cls_name='source', anomaly=label))
    references = []
    for index in range(5):
        image = f'ref_{index}.png'
        (tmp_path/image).touch()
        references.append(dict(img_path=image, mask_path='', cls_name='source', anomaly=0))
    (tmp_path/'meta.json').write_text(json.dumps(dict(train=dict(source=references), test=dict(source=rows))))
    first = source_split(tmp_path, 10, .2)
    second = source_split(tmp_path, 10, .2)
    assert first == second
    train_paths = {r['img_path'] for r in first['train_queries']}
    val_paths = {r['img_path'] for r in first['validation_queries']}
    support_paths = {r['img_path'] for r in first['validation_supports']}
    assert len(val_paths) == 2 and not train_paths & val_paths
    assert not support_paths & (val_paths | train_paths)
    bad = dict(train=dict(source=references+[rows[0]]), test=dict(source=rows))
    (tmp_path/'meta.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='overlap'):
        source_split(tmp_path, 10, .2)


def test_changed_frozen_weights_reject_checkpoint():
    with pytest.raises(ValueError, match='same frozen base weights'):
        validate_checkpoint_payload(dict(format='adaptclip-rma-v1', base_sha256='old', pretrained_sha256='clip'),
                                    'different', 'clip', {})
