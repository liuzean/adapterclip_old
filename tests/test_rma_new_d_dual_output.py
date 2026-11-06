"""Exact new-D dual routing and offline donor verification; not auto-run."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from fusion_ablation_utils import read_csv, write_csv
from rma_new_d_dual_common import CONDITIONS, DUAL, LOCAL, DualBuffer, final_check, route_dual
from rma_reliability import parse_args
from rma_utils import FIELDS


def fixture_outputs():
    raw = {'G0_L1': torch.full((1, 2, 2), .8), LOCAL: torch.full((1, 2, 2), .2)}
    maps = {n: v.clone() for n, v in raw.items()}
    scores = {'G0_L1': torch.tensor([.7]), LOCAL: torch.tensor([.1])}
    for name in ('baseline', 'G1_L1'):
        raw[name] = raw['G0_L1']; maps[name] = maps['G0_L1']; scores[name] = scores['G0_L1']
    return raw, maps, scores


def test_route_aliases_complete_final_score_and_map_without_refusion():
    raw, maps, scores = fixture_outputs()
    route_dual(raw, maps, scores)
    assert raw[DUAL] is raw[LOCAL] and maps[DUAL] is maps[LOCAL]
    assert scores[DUAL] is scores['G0_L1'] and scores[DUAL] is not scores[LOCAL]
    assert float(scores[DUAL][0]) != float(maps[LOCAL].max())


def test_buffer_rejects_recomputed_image_score_and_wrong_pixel_map():
    raw, maps, scores = fixture_outputs(); route_dual(raw, maps, scores)
    buffer = DualBuffer('example', conditions=CONDITIONS)
    assert set(buffer.maps) == {'baseline', 'G0_L1', LOCAL}
    scores[DUAL] = scores[LOCAL]
    with pytest.raises(ValueError, match='image output'):
        buffer.append(torch.zeros(2, 2), 0, maps, scores, 0)
    route_dual(raw, maps, scores); maps[DUAL] = maps['G0_L1']
    with pytest.raises(ValueError, match='pixel output'):
        buffer.append(torch.zeros(2, 2), 0, maps, scores, 0)


def test_default_scope_remains_all_shots_and_datasets():
    args = parse_args([])
    assert args.shots == [1, 2, 4] and len(args.datasets) == 6


@pytest.mark.parametrize('mode,old_name,offline_name', [('fixed125', 'G0_L1_fixed', 'dual_alpha12.5'), ('beta4', 'beta_4', 'dual_beta4')])
def test_offline_checks_use_image_and_pixel_from_correct_donors(tmp_path, mode, old_name, offline_name):
    history = tmp_path/'history'/'Visa'/'1shot_seed10'; history.mkdir(parents=True)
    output = tmp_path/'out'; output.mkdir()
    old_scores = [dict(Category='Mean', **{'Image Path': 'image.png'}, Label=1,
        **{'G0_L1 Image Score': .7, f'{old_name} Image Score': .2})]
    write_csv(history/'image_scores.csv', list(old_scores[0]), old_scores)
    fresh = [dict(Category='Mean', **{'Image Path': 'image.png'}, Label=1,
        **{'G0_L1 Image Score': .7, f'{LOCAL} Image Score': .2, f'{DUAL} Image Score': .7})]
    write_csv(output/'image_scores.csv', list(fresh[0]), fresh)
    original = [dict(Category='Mean', Condition=old_name, **{f: 50. for f in FIELDS})]
    write_csv(history/'metrics_combined.csv', list(original[0]), original)
    expected = dict(Category='Mean', Dataset='Visa', Shot=1, Candidate=offline_name,
        **{f: 90. if f.startswith('I-') else 50. for f in FIELDS})
    offline_file = tmp_path/'offline.csv'; write_csv(offline_file, list(expected), [expected])
    current = [dict(Category='Mean', Condition=n, **{f: (90. if f.startswith('I-') else 50.) if n != LOCAL else 50. for f in FIELDS})
        for n in ('G0_L1', LOCAL, DUAL)]
    args = SimpleNamespace(_dataset='Visa', _shot=1, local_history=tmp_path/'history', offline_metrics=offline_file)
    assert final_check(output, current, args, mode)['offline_combination_parity_passed']
    assert all(r['Passed'] == 'True' for r in read_csv(output/'offline_dual_metric_checks.csv'))
    current[-1]['P-AP'] += 1
    with pytest.raises(ValueError):
        final_check(output, current, args, mode)
