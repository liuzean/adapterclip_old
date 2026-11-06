"""Local error attribution contracts; authored, not automatically run."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_local_diagnostics import parse_args, selection
from rma_local_diagnostics_utils import CONDITIONS, describe, measure, regions, summaries
from fusion_ablation_utils import read_csv


def test_default_source_shots_and_fixed_selection():
    args = parse_args([])
    assert args.shots == [1, 2, 4] and args.vis_per_type == 2
    split = dict(validation_queries=[dict(cls_name='c', anomaly=a, img_path=f'{a}_{i}') for a in (0, 1) for i in range(3)])
    assert selection(split, 1) == {('c', '0_0'), ('c', '1_0')}


def test_empty_region_does_not_become_zero_score_and_nan_fails():
    assert describe(np.array([]))['Mean'] is None
    assert describe(np.array([]))['Pixels'] == 0
    with pytest.raises(ValueError):
        describe([float('nan')])
    with pytest.raises(ValueError):
        regions(np.ones((1, 1)), 0)


def test_pair_isolates_added_background_high_response_and_defect_miss():
    mask = np.array([[True, False]])
    before = np.array([[.8, .2]])
    after = np.array([[.2, .8]])
    maps = {n: before.copy() for n in CONDITIONS}; maps['G0_L1'] = after
    _, gaps, effects = measure('x', 'c', 1, mask, {'evaluated': maps}, 1)
    rows = [r for r in effects if r['Effect'] == 'add_RMA' and r['Threshold'] == .5]
    fg = next(r for r in rows if r['Region'] == 'defect')
    bg = next(r for r in rows if r['Region'] == 'anomaly_background')
    assert fg['NewMissPixels'] == 1 and fg['RecoveredPixels'] == 0
    assert bg['NewHighPixels'] == 1 and bg['RemovedHighPixels'] == 0
    assert next(r for r in gaps if r['Condition'] == 'G0_L1')['ForegroundMinusBackground'] < 0


def test_no_change_has_zero_paired_error_and_normal_has_no_defect_region(tmp_path):
    mask = np.zeros((2, 2), bool); maps = {n: np.full((2, 2), .3) for n in CONDITIONS}
    rows, gaps, effects = measure('x', 'c', 0, mask, {'evaluated': maps}, 1)
    assert not gaps and all(r['Region'] == 'normal_background' for r in rows)
    assert all(r['MeanDelta'] == 0 and r['NewHighPixels'] == r['RemovedHighPixels'] == 0 for r in effects)
    summaries(tmp_path, rows, effects, gaps)
    assert all(float(r['NetErrorRateChange']) == 0 for r in read_csv(tmp_path/'paired_error_summary.csv'))


def test_image_average_tail_is_not_silently_reported_as_pooled_quantile(tmp_path):
    rows = []; gaps = []; effects = []
    for shape, value in [((1, 1), .9), ((3, 3), .1)]:
        r, g, e = measure(str(shape), 'c', 0, np.zeros(shape, bool),
            {'evaluated': {n: np.full(shape, value) for n in CONDITIONS}}, 1)
        rows.extend(r); gaps.extend(g); effects.extend(e)
    summaries(tmp_path, rows, effects, gaps)
    one = next(r for r in read_csv(tmp_path/'region_summary.csv') if r['Category'] == 'c' and r['Condition'] == 'baseline')
    assert float(one['AverageImageP99']) == pytest.approx(.5)
    assert float(one['PooledMean']) == pytest.approx(.18)
