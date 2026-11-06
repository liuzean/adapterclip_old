"""Experiment G contracts. Authored for explicit pytest use; not auto-run."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
import rma_target_diagnostics as entry
from fusion_ablation_utils import read_csv, write_csv
from rma_target_diagnostics_utils import (CONDITIONS, harmonic_score, observe, rank_summary,
                                          ranking_reports, score_index, selected_keys)


def test_default_schedule_and_cwd_independent_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = entry.parse_args([])
    assert args.datasets == ['SDD', 'part_glass'] and args.shots == [1, 2, 4]
    assert args.num_workers == 0 and args.save_maps is False
    assert args.e_run == entry.ROOT/'results/rma_fixed_dual_output/six_datasets_seed10'
    command = entry.child_command(args, tmp_path/'run', tmp_path/'out', 'SDD', 2)
    child = entry.parse_args(command[2:])
    assert child._dataset == 'SDD' and child._shot == 2 and child.num_workers == 0
    assert child.save_visualizations and not child.save_maps
    with pytest.raises(SystemExit):
        entry.parse_args(['--shots', '4', '1'])


def test_pair_auc_treats_ties_as_half_and_single_label_as_undefined():
    rows = [dict(Label=0, Score=.2), dict(Label=0, Score=.4), dict(Label=1, Score=.4)]
    result = rank_summary(rows, 'Score')
    assert result['AUROC'] == 75 and result['CorrectPairs'] == result['TiedPairs'] == 1
    assert rank_summary(rows[:2], 'Score')['AUROC'] is None
    with pytest.raises(ValueError):
        rank_summary([dict(Label=1, Score=float('nan'))], 'Score')
    with pytest.raises(ValueError):
        rank_summary([dict(Label=2, Score=.2)], 'Score')


def test_duplicate_scores_are_rejected_instead_of_overwritten(tmp_path):
    row = dict(Category='c', **{'Image Path': str(tmp_path/'x.png')})
    with pytest.raises(ValueError, match='Duplicate'):
        score_index([row, dict(row)])


def test_fixed_selection_uses_manifest_order_and_both_labels(tmp_path):
    rows = [dict(Category='c', **{'Sample Type': kind, 'Image Path': str(tmp_path/f'{kind}_{i}.png')})
            for kind in ('Normal', 'Anomaly') for i in range(3)]
    selected = selected_keys(rows, 1)
    assert len(selected) == 2
    assert all('_0.png' in path for _, path in selected)
    assert not selected_keys(rows, 0)


def test_region_stats_distinguish_new_background_error_from_defect_miss():
    good = np.array([[.8, .2]], dtype=np.float32)
    bad = np.array([[.2, .8]], dtype=np.float32)
    maps = {name: good for name in CONDITIONS}
    maps['E'] = bad
    mask = np.array([[True, False]])
    rows, gaps, effects = observe('x', 'c', 1, mask, {'evaluated': maps}, 1)
    fg = next(row for row in effects if row['Effect'] == 'candidate_vs_off' and row['Region'] == 'defect' and row['Threshold'] == .5)
    bg = next(row for row in effects if row['Effect'] == 'candidate_vs_off' and row['Region'] == 'anomaly_background' and row['Threshold'] == .5)
    assert fg['NewMissPixels'] == 1 and bg['NewHighPixels'] == 1
    assert next(row for row in gaps if row['Condition'] == 'E')['ForegroundMinusBackground'] < 0
    assert all(row['Pixels'] == 1 for row in rows)
    with pytest.raises(ValueError):
        observe('x', 'c', 0, mask, {'evaluated': maps}, 1)


def test_ranking_report_identifies_the_actual_newly_inverted_pair(tmp_path):
    rows = []
    for label, image in ((0, 'normal.png'), (1, 'defect.png')):
        row = dict(Category='c', Label=label, **{'Image Path': image})
        row.update({f'{name} Image Score': .8 if label else .2 for name in CONDITIONS})
        row['E Image Score'] = .1 if label else .9
        rows.append(row)
    summaries = ranking_reports(tmp_path, rows, [])
    assert next(row for row in summaries if row['Condition'] == 'A')['AUROC'] == 100
    assert next(row for row in summaries if row['Condition'] == 'E')['AUROC'] == 0
    changed = [row for row in read_csv(tmp_path/'ranking_changed_pairs.csv') if row['Effect'] == 'candidate_vs_A']
    assert len(changed) == 1 and float(changed[0]['PairCreditDelta']) == -1
    assert changed[0]['AnomalyImage'] == 'defect.png' and changed[0]['NormalImage'] == 'normal.png'


def test_score_reconstruction_requires_the_scoring_map_maximum():
    # E uses the donor25% maximum for its image score, not its displayed12.5% map.
    native, scoring_maximum, displayed_maximum = .8, .2, .4
    assert harmonic_score(native, scoring_maximum) == pytest.approx(.32)
    assert harmonic_score(native, displayed_maximum) != pytest.approx(.32)
    assert harmonic_score(0., 0.) == 0
    with pytest.raises(ValueError):
        harmonic_score(.8, float('inf'))


def test_historical_reference_order_mismatch_stops_before_initialization(monkeypatch, tmp_path):
    args = SimpleNamespace(ablation_run=tmp_path/'ablation', three_run=tmp_path/'three', e_run=tmp_path/'e')
    manifest = [{'Category': 'c', 'Image Path': 'query.png', 'Sample Type': 'Normal'}]
    source = dict(manifest=manifest, reference_paths=[('c', 'r1.png'), ('c', 'r2.png')])
    monkeypatch.setattr(entry, 'target_inputs', lambda *args: (source, None, {}))
    folder = args.ablation_run/'SDD/2shot_seed10'
    folder.mkdir(parents=True)
    (folder/'run_status.json').write_text(json.dumps(dict(complete=True)))
    (folder/'input_audit.json').write_text('{}')
    write_csv(folder/'sample_manifest.csv', list(manifest[0]), manifest)
    write_csv(folder/'references.csv', ['Category', 'Image Path'],
              [{'Category': 'c', 'Image Path': path} for path in ('r2.png', 'r1.png')])
    with pytest.raises(ValueError, match='input mismatch'):
        entry.history_round(args, 'SDD', 2, 'new', 'three')


def test_changed_history_or_weight_cannot_be_silently_resumed(monkeypatch, tmp_path):
    path = tmp_path/'weight.pth'
    path.write_bytes(b'previous checkpoint')
    monkeypatch.setattr(entry, 'code_hashes', lambda: {})
    lock = dict(implementation_sha256={}, historical_sha256={str(path): entry.digest(path)})
    entry.verify_lock(lock)
    path.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='changed'):
        entry.verify_lock(lock)


def test_status_only_never_launches_a_worker(monkeypatch, tmp_path, capsys):
    (tmp_path/'run_status.json').write_text(json.dumps(dict(phase='diagnostic_inference', current='SDD/2', complete=False)))
    def forbidden(*args):
        raise AssertionError('Read-only --status launched work')
    monkeypatch.setattr(entry, 'preflight', forbidden)
    monkeypatch.setattr(entry, 'run_child', forbidden)
    entry.main(['--status', str(tmp_path)])
    assert 'SDD/2' in capsys.readouterr().out
