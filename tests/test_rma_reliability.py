"""Meaningful formula/isolation/selection contracts; NOT auto-executed."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_reliability import parse_args
from rma_reliability_utils import (BETAS, CANDIDATE, FIELDS, ReliabilityBuffer,
                                   dispersion_map, gated_output, select_beta, summarize)
from fusion_ablation_utils import read_csv, read_json, write_csv
from rma_output_ablation_utils import route_outputs


def matching_features(ambiguous=False):
    query = torch.zeros(1, 17, 768)
    query[:, 1:, :8] = 1/(8**.5)
    support = torch.zeros(1, 1, 17, 768)
    if ambiguous:
        support[0, 0, 1:9, :8] = torch.eye(8)
        support[0, 0, 9:, :8] = -torch.eye(8)
    else:
        support[:, :, 1:, :8] = 1/(8**.5)
    return [query.clone() for _ in range(4)], [support.clone() for _ in range(4)]


def outputs():
    from adaptcliplib import fusion_fun
    mean3 = torch.full((1, 4, 4), .2)
    align = torch.full_like(mean3, .6)
    old = dict(no_align=mean3, baseline=fusion_fun([mean3, align], fusion_type='harmonic_mean'))
    native = torch.tensor([.4])
    maps = dict(baseline=old['baseline'].clone())
    scores = dict(baseline=fusion_fun([native, maps['baseline'].flatten(1).max(1).values], fusion_type='harmonic_mean'))
    g = torch.tensor([[0., 3.]])
    l = g[:, :, None, None].expand(1, 2, 4, 4).clone()
    extra = dict(ALIGN=align)
    raw, evaluated, image, _ = route_outputs(old, maps, scores, native, extra, g, l, 0)
    return old, native, extra, l, (raw['G0_L1'], evaluated['G0_L1'], image['G0_L1'])


def test_default_all_conditions_and_no_user_target_tuning_parameter():
    args = parse_args([])
    assert args.datasets == ['Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass']
    assert args.shots == [1, 2, 4] and not hasattr(args, 'beta') and not hasattr(args, 'seed')


def test_one_shot_cohesion_is_not_reference_image_count():
    coherent = dispersion_map(*matching_features(), 4, chunk=3)
    ambiguous = dispersion_map(*matching_features(True), 4, chunk=3)
    assert torch.allclose(coherent, torch.zeros_like(coherent), atol=1e-6)
    assert torch.allclose(ambiguous, torch.ones_like(ambiguous), atol=1e-6)


def test_dispersion_has_no_cross_query_batch_dependency_and_chunk_drift():
    q, r = matching_features(True)
    before = [v.clone() for v in (*q, *r)]
    single = dispersion_map(q, r, 4, chunk=3)
    q2, r2 = matching_features(False)
    batched = dispersion_map([torch.cat(x) for x in zip(q, q2)], [torch.cat(x) for x in zip(r, r2)], 4, chunk=16)
    assert torch.allclose(single, batched[:1], atol=1e-6)
    assert all(torch.equal(x, y) for x, y in zip(before, (*q, *r)))


def test_degenerate_or_nan_features_fail():
    q, r = matching_features()
    q[0][:, 1:] = 0
    with pytest.raises(ValueError, match='degenerate'):
        dispersion_map(q, r, 4)
    q, r = matching_features()
    r[0][:, :, 1, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        dispersion_map(q, r, 4)


def test_zero_strength_is_exact_G0_L1_including_final_image_score():
    old, native, extra, l, endpoint = outputs()
    raw, maps, image, alpha = gated_output(old, native, extra, l, torch.ones(1, 4, 4), 0, 0, endpoint)
    assert raw is endpoint[0] and maps is endpoint[1] and image is endpoint[2]
    assert torch.equal(alpha, torch.full_like(alpha, .25))


def test_gate_changes_only_local_mixture_before_align_and_keeps_G0():
    from adaptcliplib import fusion_fun
    old, native, extra, l, endpoint = outputs()
    d = torch.ones(1, 4, 4)
    raw, maps, image, alpha = gated_output(old, native, extra, l, d, 4, 0, endpoint)
    expected = fusion_fun([(1-alpha)*old['no_align']+alpha*l.softmax(1)[:, 1], extra['ALIGN']], fusion_type='harmonic_mean')
    assert torch.equal(raw, expected) and torch.all(alpha > 0) and torch.all(alpha < .25)
    expected_image = fusion_fun([native, maps.flatten(1).max(1).values], fusion_type='harmonic_mean')
    assert torch.equal(image, expected_image) and torch.all(image < endpoint[2])
    assert torch.equal(native, torch.tensor([.4]))


def test_buffer_stores_only_three_unique_target_maps():
    buffer = ReliabilityBuffer('example')
    assert set(buffer.maps) == {'baseline', 'G0_L1', CANDIDATE}


def test_source_selection_combines_all_shots_and_ties_choose_zero():
    tables = {s: [dict(Category='Mean', Condition='G0_L1' if b == 0 else f'beta_{int(b)}',
                      **{f: 90. for f in FIELDS}) for b in BETAS] for s in (1, 2, 4)}
    assert select_beta(tables)[0] == 0
    tables[4][2]['P-AP'] += 1
    assert select_beta(tables)[0] == 16


def test_overall_requires_two_shots_and_cannot_select_comparator_afterwards(tmp_path):
    datasets = ['Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass']
    jobs = []
    for d in datasets:
        for s in (1, 2, 4):
            directory = tmp_path/d/f'{s}shot_seed10'; directory.mkdir(parents=True)
            rows = [dict(Category='Mean', Condition=n, **{f: 50. for f in FIELDS})
                    for n in ('baseline', 'G0_L1', 'G1_L1', CANDIDATE)]
            # Only 1-shot improves against direct G0_L1. All improve against A.
            for row in rows:
                if row['Condition'] == CANDIDATE:
                    row.update({f: 51. if s == 1 else 49. for f in FIELDS})
                elif row['Condition'] == 'baseline':
                    row.update({f: 40. for f in FIELDS})
            write_csv(directory/'metrics_combined.csv', list(rows[0]), rows); jobs.append((d, s))
    summarize(tmp_path, jobs, datasets)
    overall = read_json(tmp_path/'retention_overall.json')
    primary = next(r for r in overall if r['Primary'])
    assert primary['Comparator'] == 'G0_L1' and primary['PassingShots'] == 1 and primary['Retain'] is False
    assert next(r for r in overall if r['Comparator'] == 'baseline')['Retain'] is True
    assert len(read_csv(tmp_path/'metrics_summary.csv')) == 72
