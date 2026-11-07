"""H real-inference contracts; authored for explicit testing, not auto-run."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
import rma_score_coordination_inference as entry
from fusion_ablation_utils import write_csv
from rma_score_coordination_inference_utils import LOCAL, HBuffer, HRouter, final_check
from rma_utils import FIELDS


def test_default_schedule_paths_and_no_new_matrices(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = entry.parse_args([])
    assert args.shots == [1, 2, 4] and len(args.datasets) == 6
    assert args.h_run == entry.ROOT/'results/rma_score_coordination/experiment_H_seed10'
    assert not args.save_maps and not args.save_visualizations
    with pytest.raises(ValueError):
        entry.parse_args(['--resume_from', 'x', '--save_maps'])


def test_router_uses_actual_G0_and_complete_E_score_and_aliases_pixel_map():
    history = [dict(Category='c', **{'Image Path': 'q.png'}, Label=1, E=.2, G0=.8, H=.35)]
    router = HRouter(history); router.native = torch.tensor([.8])
    local = torch.full((1, 2, 2), .9)
    raw = {LOCAL: local}; maps = {LOCAL: local.clone()}
    scores = {'G0_L1': torch.tensor([.2]), LOCAL: torch.tensor([.99])}
    router.route(raw, maps, scores)
    assert maps['H'] is maps['E'] is maps[LOCAL]
    assert raw['H'] is raw['E'] is raw[LOCAL]
    assert scores['E'] is scores['G0_L1']
    assert float(scores['H'][0]) == pytest.approx(.35)
    assert router.native is None and router.rows[0]['FinalPixelEqual']
    router = HRouter(history); router.native = torch.tensor([.1])
    with pytest.raises(ValueError, match='offline H'):
        router.route(raw, maps, scores)


def test_buffer_evaluates_H_image_scores_but_reuses_equal_E_pixel_metrics():
    buffer = HBuffer('c')
    assert len(buffer.maps) == 3
    for index, label in enumerate((0, 1)):
        maps = {name: torch.full((1, 2, 2), .1 if label == 0 else .8) for name in ('baseline', 'G0_L1', LOCAL)}
        maps['G1_L1'] = maps['G0_L1']; maps['E'] = maps[LOCAL]; maps['H'] = maps['E']
        scores = {name: torch.tensor([.8 if label == 0 else .2]) for name in ('baseline', 'G0_L1', 'G1_L1', LOCAL, 'E')}
        scores['H'] = torch.tensor([.2 if label == 0 else .8])
        mask = torch.full((2, 2), label, dtype=torch.uint8)
        buffer.append(mask, label, maps, scores, 0)
    pixels, images = buffer.finish('cpu')
    h = next(row for row in images if row['Condition'] == 'H')
    e = next(row for row in images if row['Condition'] == 'E')
    assert h['AUROC'] == 100 and e['AUROC'] == 0
    hp = next(row for row in pixels if row['Condition'] == 'H')
    ep = next(row for row in pixels if row['Condition'] == 'E')
    assert all(hp[field] == ep[field] for field in ('AUROC', 'AP', 'F1max'))


def test_final_check_rejects_offline_metric_mismatch_and_wrong_query_order(tmp_path):
    output, history, e_dir = [tmp_path/name for name in ('out', 'h', 'e')]
    for folder in (output, history, e_dir):
        folder.mkdir()
    score = dict(Category='c', **{'Image Path': str(tmp_path/'q.png')}, Label=1)
    write_csv(output/'image_scores.csv', list(score), [score])
    router = HRouter([score]); router.rows = [dict(score, G0=.8, E=.2, H=.35, RawPixelEqual=True, FinalPixelEqual=True)]
    table = [dict(Category='c', Condition=name, **{field: 80. for field in FIELDS}) for name in ('H', 'E')]
    bad = [dict(table[0], **{'I-AUROC': 79.})]
    write_csv(history/'metrics_combined.csv', list(bad[0]), bad)
    e = dict(table[1], Condition='dual_output')
    write_csv(e_dir/'metrics_combined.csv', list(e), [e])
    with pytest.raises(ValueError, match='six metrics differ'):
        final_check(output, table, router, history, e_dir)
    router.rows[0]['Image Path'] = str(tmp_path/'different.png')
    with pytest.raises(ValueError, match='identity/order'):
        final_check(output, table, router, history, e_dir)


def test_status_does_not_start_inference(monkeypatch, tmp_path, capsys):
    entry.write_json(tmp_path/'run_status.json', dict(complete=False, current='Visa/2'))
    def forbidden(*args):
        raise AssertionError('Read-only status launched work')
    monkeypatch.setattr(entry, 'preflight', forbidden)
    monkeypatch.setattr(entry, 'run_child', forbidden)
    entry.main(['--status', str(tmp_path)])
    assert 'Visa/2' in capsys.readouterr().out
