"""Constant-fraction arithmetic and source-selection contracts; not auto-run."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_fixed_fusion import command
from rma_fixed_fusion_utils import CHOICES, dispersion_map, gated_output, select_fraction
from rma_output_ablation_utils import route_outputs
from rma_reliability import parse_args
from rma_utils import FIELDS


def fixture():
    from adaptcliplib import fusion_fun
    base = torch.full((1, 2, 2), .2)
    align = torch.full_like(base, .6)
    old = dict(no_align=base, baseline=fusion_fun([base, align], fusion_type='harmonic_mean'))
    fixed = torch.tensor([.3]); extra = dict(ALIGN=align)
    maps = dict(baseline=old['baseline'].clone())
    scores = dict(baseline=fusion_fun([fixed, maps['baseline'].flatten(1).max(1).values], fusion_type='harmonic_mean'))
    g = torch.tensor([[0., 1.]])
    l = torch.tensor([0., 2.]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2).clone()
    raw, evaluated, image, _ = route_outputs(old, maps, scores, fixed, extra, g, l, 0)
    return old, fixed, extra, l, (raw['G0_L1'], evaluated['G0_L1'], image['G0_L1']), scores


def test_fixed_control_does_not_retrieve_or_use_spatial_dispersion():
    assert dispersion_map(None, None, 518) is None


def test_quarter_exactly_aliases_existing_G0_L1():
    old, fixed, extra, l, endpoint, _ = fixture()
    raw, maps, score, alpha = gated_output(old, fixed, extra, l, None, .25, 0, endpoint)
    assert raw is endpoint[0] and maps is endpoint[1] and score is endpoint[2]
    assert torch.equal(alpha, torch.full_like(alpha, .25))


def test_zero_restores_three_branch_map_and_image_scoring():
    old, fixed, extra, l, endpoint, scores = fixture()
    raw, _, score, alpha = gated_output(old, fixed, extra, l, None, 0, 0, endpoint)
    assert raw is old['baseline'] and torch.equal(score, scores['baseline'])
    assert torch.count_nonzero(alpha) == 0


def test_fixed_ratio_is_applied_before_align_and_own_map_max():
    from adaptcliplib import fusion_fun
    old, fixed, extra, l, endpoint, _ = fixture()
    for ratio in (.125, .375):
        raw, maps, score, alpha = gated_output(old, fixed, extra, l, None, ratio, 0, endpoint)
        expected = fusion_fun([(1-ratio)*old['no_align']+ratio*l.softmax(1)[:, 1], extra['ALIGN']], fusion_type='harmonic_mean')
        assert torch.equal(raw, expected) and torch.all(alpha == ratio)
        assert torch.equal(score, fusion_fun([fixed, maps.flatten(1).max(1).values], fusion_type='harmonic_mean'))


def test_source_ties_keep_quarter_and_one_fraction_shared_across_shots():
    tables = {s: [dict(Category='Mean', Condition=n, **{f: 90. for f in FIELDS}) for _, n in CHOICES] for s in (1, 2, 4)}
    assert select_fraction(tables)[0] == .25
    tables[4][2]['P-AP'] += 1
    assert select_fraction(tables)[0] == .125


def test_child_uses_fixed_entry_and_default_scope():
    args = parse_args([])
    assert args.shots == [1, 2, 4] and len(args.datasets) == 6
    values = dict(vars(args), save_maps=False)
    cmd = command(SimpleNamespace(**values), Path('example'), 'calibrate', 1)
    assert Path(cmd[2]).name == 'rma_fixed_fusion.py' and '--_phase' in cmd
