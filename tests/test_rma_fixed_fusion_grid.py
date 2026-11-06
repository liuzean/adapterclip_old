"""Offline fraction-grid assembly and retention; authored, not auto-executed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from fusion_ablation_utils import read_csv, read_json, write_csv
from rma_fixed_fusion_grid import DATASETS, FIELDS, merge


def test_reuse_and_new_fraction_remain_distinct_and_require_two_shots(tmp_path):
    reused = []; jobs = []
    for dataset in DATASETS:
        for shot in (1, 2, 4):
            for fraction in (0., .25, .375):
                reused.append(dict(Dataset=dataset, Shot=shot, Fraction=fraction, HistoricalCondition='old', Origin='history',
                    Category='Mean', Condition='old', **{f: 50. for f in FIELDS}))
            directory = tmp_path/dataset/f'{shot}shot_seed10'; directory.mkdir(parents=True)
            row = dict(Category='Mean', Condition='G0_L1_fixed', **{f: 51. if shot == 2 else 49. for f in FIELDS})
            write_csv(directory/'metrics_combined.csv', list(row), [row]); jobs.append((dataset, shot))
    merge(tmp_path, reused, jobs)
    rows = read_csv(tmp_path/'all_fraction_metrics.csv')
    assert len(rows) == 72 and {r['Condition'] for r in rows} == {'fraction_0', 'fraction_125', 'fraction_250', 'fraction_375'}
    overall = read_json(tmp_path/'all_fraction_retention_overall.json')
    candidate = next(r for r in overall if r['Fraction'] == .125)
    assert candidate['All18Complete'] and candidate['PassingShots'] == 1 and candidate['RetainIncrement'] is False


def test_missing_new_fraction_is_not_reported_as_complete(tmp_path):
    reused = [dict(Dataset=d, Shot=s, Fraction=fraction, HistoricalCondition='old', Origin='history', Category='Mean',
        Condition='old', **{f: 50. for f in FIELDS}) for d in DATASETS for s in (1, 2, 4) for fraction in (0., .25, .375)]
    merge(tmp_path, reused, [])
    candidate = next(r for r in read_json(tmp_path/'all_fraction_retention_overall.json') if r['Fraction'] == .125)
    assert candidate['All18Complete'] is False and candidate['RetainIncrement'] is False
