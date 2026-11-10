"""Experiment J CPU regression contracts; authored, not automatically executed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_align_ablation import parse_args
from rma_align_ablation_utils import CONDITIONS, DATASETS, comparison_tables, local_outputs, retention
from rma_utils import FIELDS


def test_only_local_ALIGN_changes_and_same_smoothing_is_used(monkeypatch):
    import branch_diagnostics_utils
    calls = []
    original = branch_diagnostics_utils.smooth_maps
    def spy(maps, sigma):
        calls.append((maps.clone(), sigma))
        return original(maps, sigma)
    monkeypatch.setattr(branch_diagnostics_utils, 'smooth_maps', spy)
    three = torch.linspace(.05, .8, 18).reshape(2, 3, 3)
    logits = torch.arange(36).float().reshape(2, 2, 3, 3)/20
    a = local_outputs(three, logits, torch.full_like(three, .1), 4)
    b = local_outputs(three, logits, torch.full_like(three, .9), 4)
    assert [sigma for _, sigma in calls] == [4, 4, 4, 4]
    assert torch.equal(a[0]['J_no_ALIGN'], b[0]['J_no_ALIGN'])
    assert torch.equal(a[1]['J_no_ALIGN'], b[1]['J_no_ALIGN'])
    assert not torch.equal(a[1]['H'], b[1]['H'])
    assert all(m.shape == three.shape for m in a[1].values())
    with pytest.raises(ValueError):
        local_outputs(three, logits, torch.ones_like(three), 0)


def test_one_forward_matches_existing_H_and_never_rescores_no_ALIGN(monkeypatch):
    import fusion_ablation
    import test as original_test
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import smooth_maps
    from h_inference import predict as original_h
    from rma_align_ablation_utils import predict
    from rma_output_ablation_utils import route_outputs
    calls = dict(encoder=0, adapters=0, rma=0)
    three = torch.linspace(.1, .8, 18).reshape(2, 3, 3)
    align = torch.full_like(three, .08)
    native = torch.tensor([.8, .6])
    raw = dict(no_align=three, baseline=fusion_fun([three, align], fusion_type='harmonic_mean'))
    old_maps = dict(baseline=smooth_maps(raw['baseline'], 4))
    old_scores = dict(baseline=torch.tensor([.2, .25]))
    features, patches = torch.zeros(2, 8), [torch.zeros(2, 10, 8)]*4
    pf, pp = torch.zeros(2, 1, 8), [torch.zeros(2, 1, 10, 8)]*4
    global_logits, local_logits = torch.tensor([[.2, .8], [.8, .2]]), torch.zeros(2, 2, 3, 3)
    def encode(*args, **kwargs):
        calls['encoder'] += 1
        return features, patches
    def infer(*args, **kwargs):
        calls['adapters'] += 1
        return raw, old_maps, old_scores, {}, {}, native, dict(ALIGN=align)
    def rma(*args):
        calls['rma'] += 1
        return global_logits, local_logits
    monkeypatch.setattr(fusion_ablation, 'infer_batch', infer)
    monkeypatch.setattr(original_test, 'prompt_association', lambda *args: (pf, pp))
    state = SimpleNamespace(model=SimpleNamespace(encode_image=encode), device='cpu',
        config=SimpleNamespace(features_list=[6, 12, 18, 24], sigma=4), memory=(None, None),
        adapters=(None, None, None), text=None, rma=rma)
    images = torch.zeros(2, 3, 3, 3)
    maps, scores, parts = predict(state, images, ['a', 'b'], ['a.png', 'b.png'])
    assert calls == dict(encoder=1, adapters=1, rma=1)
    assert scores['H'] is scores['J_no_ALIGN']
    assert torch.equal(scores['H'], .75*parts['E']+.25*parts['G0'])
    _, _, routed, _ = route_outputs(raw, old_maps, old_scores, native, dict(ALIGN=align), global_logits, local_logits, 4)
    assert torch.equal(parts['E'], routed['G0_L1'])
    rescored = fusion_fun([native, maps['J_no_ALIGN'].flatten(1).max(1).values], fusion_type='harmonic_mean')
    assert not torch.allclose(scores['H'], .75*rescored+.25*native)
    pixel, score, _ = original_h(state, images, ['a', 'b'], ['a.png', 'b.png'])
    assert torch.equal(maps['H'], pixel) and torch.equal(scores['H'], score)


def test_default_schedule_and_resume_options_are_locked():
    args = parse_args([])
    assert args.datasets == list(DATASETS) and args.shots == [1, 2, 4] and args.num_workers == 0
    assert parse_args(['--resume_from', 'results/rma_align_ablation/J_seed10']).resume_from
    for argv in (['--shots', '4', '1'], ['--datasets', 'SDD', 'Visa'],
                 ['--resume_from', 'results/rma_align_ablation/J_seed10', '--shots', '1']):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_full_six_metrics_history_parity_and_image_changes_are_detected():
    def row(category, condition, value):
        return dict(Category=category, Condition=condition, **{f: value for f in FIELDS})
    history = [row(c, n, 50) for c in ('a', 'Mean') for n in ('H', 'baseline')]
    current = [row(c, n, 50) for c in ('a', 'Mean') for n in CONDITIONS]
    for r in current:
        if r['Condition'] == 'J_no_ALIGN':
            r['P-AP'] += 1
    combined, parity, identity, deltas = comparison_tables(current, history, ['a'])
    assert len(combined) == 6 and all(r['Passed'] for r in (*parity, *identity))
    assert all(r['ImageNet'] == 0 and r['PixelNet'] == 1 for r in deltas)
    current[0]['I-AP'] += .1
    _, parity, identity, _ = comparison_tables(current, history, ['a'])
    assert not all(r['Passed'] for r in parity) and not all(r['Passed'] for r in identity)
    with pytest.raises(ValueError):
        comparison_tables(current[:-1], history, ['a'])


def test_retention_requires_two_complete_shots_and_strictly_positive_net():
    def records(shots):
        values = (1, 1, 1, -1, -1, -.5)
        return [dict(Category='Mean', Comparator=c, Dataset=d, Shot=s, Net=v)
                for s in shots for c in ('H', 'baseline') for d, v in zip(DATASETS, values)]
    _, single = retention(records([1]))
    assert all(not r['retain'] for r in single.values())
    _, full = retention(records([1, 2, 4]))
    assert all(r['retain'] for r in full.values())
    tied = records([1, 2, 4])
    for r in tied:
        if r['Dataset'] == 'part_glass':
            r['Net'] = -1
    rows, overall = retention(tied)
    assert not any(r['Pass'] for r in rows) and not any(r['retain'] for r in overall.values())
