"""H contracts, authored for explicit testing; not executed during development."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
import rma_score_coordination as entry
from rma_score_coordination_utils import IMAGE_FIELDS, LAMBDAS, image_metrics, mix_scores, source_selection


def tables(scores):
    return {shot: [dict(Category='Mean', Lambda=value, **{field: scores[value] for field in IMAGE_FIELDS})
                   for value in LAMBDAS] for shot in (1, 2, 4)}


def test_default_all_targets_and_shots_and_root_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = entry.parse_args([])
    assert args.shots == [1, 2, 4]
    assert args.datasets == ['Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass']
    assert args.e_run == entry.ROOT/'results/rma_fixed_dual_output/six_datasets_seed10'
    with pytest.raises(SystemExit):
        entry.parse_args(['--resume_from', 'x', '--shots', '1'])


def test_mixing_restores_exact_E_endpoint_and_native_endpoint():
    e = torch.tensor([.2, .8]); native = torch.tensor([.6, .4])
    assert mix_scores(e, native, 0) is e
    assert mix_scores(e, native, 1) is native
    assert torch.allclose(mix_scores(e, native, .5), torch.tensor([.4, .6]))
    with pytest.raises(ValueError):
        mix_scores(e, torch.tensor([float('nan'), .4]), .5)
    with pytest.raises(ValueError):
        mix_scores(e, native, .3)


def test_source_tie_preserves_E_and_one_global_lambda_selected():
    values = {value: 80. for value in LAMBDAS}
    assert source_selection(tables(values))[0] == 0
    values[.5] = 81.
    assert source_selection(tables(values))[0] == .5
    broken = tables(values); del broken[4]
    with pytest.raises(ValueError):
        source_selection(broken)


def test_image_metrics_keep_macro_averaging_and_original_trapezoidal_AP():
    rows = [dict(Category='small', Label=0, Score=.2), dict(Category='small', Label=1, Score=.8),
            dict(Category='large', Label=0, Score=.4), dict(Category='large', Label=0, Score=.4),
            dict(Category='large', Label=1, Score=.4)]
    values = image_metrics(rows, 'Score', 'H')
    small, large, mean = [next(row for row in values if row['Category'] == name) for name in ('small', 'large', 'Mean')]
    assert small['I-AUROC'] == 100 and large['I-AUROC'] == 50 and mean['I-AUROC'] == 75
    # A tied PR curve has trapezoidal area (1 + prevalence)/2, not AP=prevalence.
    assert large['I-AP'] == pytest.approx(100*(1+1/3)/2, abs=.001)


def test_invalid_source_metric_or_duplicate_mean_is_not_selected():
    values = {value: 80. for value in LAMBDAS}; bad = tables(values)
    bad[1].append(dict(bad[1][0]))
    with pytest.raises(ValueError):
        source_selection(bad)
    bad = tables(values); bad[1][0]['I-AP'] = float('nan')
    with pytest.raises(ValueError):
        source_selection(bad)


def test_status_never_runs_source_or_targets(monkeypatch, tmp_path, capsys):
    entry.write_json(tmp_path/'run_status.json', dict(complete=False, phase='target', current='target/DTD/2'))
    def forbidden(*args):
        raise AssertionError('Status launched work')
    monkeypatch.setattr(entry, 'preflight', forbidden)
    monkeypatch.setattr(entry, 'run_child', forbidden)
    entry.main(['--status', str(tmp_path)])
    assert 'target/DTD/2' in capsys.readouterr().out
