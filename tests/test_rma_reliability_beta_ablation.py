"""Multi-beta gate and independent retention contracts; not auto-executed."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from fusion_ablation_utils import read_csv, read_json, write_csv
from rma_reliability_beta_ablation import CONDITIONS, GRID, experiment, summarize_grid
from rma_reliability_utils import gated_output
from rma_utils import FIELDS


def test_three_betas_are_predeclared_and_share_same_G0_L1_comparator():
    plan = experiment()
    assert plan.CANDIDATE_PARAMETERS == {'beta_4': 4., 'beta_16': 16., 'beta_64': 64.}
    assert plan.PARAMETER == 'beta' and 'G0_L1' in plan.CONDITIONS


def test_each_beta_keeps_own_gate_and_never_exceeds_quarter():
    from adaptcliplib import fusion_fun
    mean = torch.full((1, 2, 2), .2)
    extra = {'ALIGN': torch.full_like(mean, .6)}
    old = {'no_align': mean, 'baseline': fusion_fun([mean, extra['ALIGN']], fusion_type='harmonic_mean')}
    local = torch.tensor([0., 1.]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2).clone()
    dispersion = torch.tensor([[[0., .05], [.1, .2]]])
    saved = dispersion.clone(); fractions = []
    for beta in GRID.values():
        raw, evaluated, score, alpha = gated_output(old, torch.tensor([.4]), extra, local, dispersion, beta, 0, (None, None, None))
        assert torch.isfinite(raw).all() and torch.isfinite(evaluated).all() and torch.isfinite(score).all()
        assert torch.all(alpha <= .25) and torch.all(alpha >= 0) and alpha[0, 0, 0] == .25
        fractions.append(alpha)
    assert torch.all(fractions[0] >= fractions[1]) and torch.all(fractions[1] >= fractions[2])
    assert torch.equal(dispersion, saved)


def test_summary_does_not_pool_passing_shots_across_betas(tmp_path):
    datasets = ['Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass']; jobs = []
    for dataset in datasets:
        for shot in (1, 2, 4):
            rows = []
            for name in CONDITIONS:
                value = 50.
                if name == 'beta_4':
                    value = 51. if shot in (1, 2) else 49.
                elif name == 'beta_16':
                    value = 51. if shot == 4 else 49.
                elif name == 'beta_64':
                    value = 49.
                rows.append(dict(Category='Mean', Condition=name, **{f: value for f in FIELDS}))
            directory = tmp_path/dataset/f'{shot}shot_seed10'; directory.mkdir(parents=True)
            write_csv(directory/'metrics_combined.csv', list(rows[0]), rows); jobs.append((dataset, shot))
    summarize_grid(tmp_path, jobs)
    primary = {r['Beta']: r for r in read_json(tmp_path/'all_beta_retention_overall.json') if r['Primary']}
    assert primary[4.]['Retain'] is True and primary[4.]['PassingShots'] == 2
    assert primary[16.]['Retain'] is False and primary[16.]['PassingShots'] == 1
    assert primary[64.]['Retain'] is False and primary[64.]['PassingShots'] == 0
    assert len(read_csv(tmp_path/'metrics_summary.csv')) == 108
    assert {r['Beta'] for r in read_csv(tmp_path/'deltas_vs_G0_L1.csv')} == {'4.0', '16.0', '64.0'}
