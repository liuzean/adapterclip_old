"""Dual-output contracts for explicit future execution; no automatic GPU tests."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_dual_output_utils import DualOutputBuffer, check_history, route_outputs
from rma_utils import MetricsBuffer


def test_dual_uses_original_final_image_score_and_complete_B_map():
    # Deliberately disagreeing A/B values catch accidental use of B's image score.
    raw = dict(baseline=torch.zeros(1, 2, 2), rma=torch.full((1, 2, 2), .8))
    maps = dict(baseline=torch.full((1, 2, 2), .1), rma=torch.full((1, 2, 2), .7))
    scores = dict(baseline=torch.tensor([.2]), rma=torch.tensor([.9]))
    route_outputs(raw, maps, scores)
    buffer = DualOutputBuffer('category')
    buffer.append(torch.zeros(2, 2), 0, maps, scores, 0)
    assert maps['dual_output'] is maps['rma'] and raw['dual_output'] is raw['rma']
    assert scores['dual_output'] is scores['baseline']
    assert set(buffer.maps) == {'baseline', 'rma'}  # Avoid a third full-map buffer.
    scores['dual_output'] = scores['rma']
    with pytest.raises(RuntimeError, match='image score'):
        buffer.append(torch.zeros(2, 2), 0, maps, scores, 0)


def test_misrouted_pixel_map_is_rejected():
    maps = dict(baseline=torch.zeros(1, 2, 2), rma=torch.ones(1, 2, 2), dual_output=torch.zeros(1, 2, 2))
    scores = dict(baseline=torch.tensor([.2]), rma=torch.tensor([.9]), dual_output=torch.tensor([.2]))
    with pytest.raises(RuntimeError, match='pixel map'):
        DualOutputBuffer('category').append(torch.zeros(2, 2), 0, maps, scores, 0)


def test_dual_metrics_reuse_current_A_image_and_B_pixel_evaluations(monkeypatch):
    def finish(self, device):
        return ([dict(Category='c', Condition='baseline', AUROC=20.), dict(Category='c', Condition='rma', AUROC=80.)],
                [dict(Category='c', Condition='baseline', AUROC=90.), dict(Category='c', Condition='rma', AUROC=30.)])
    monkeypatch.setattr(MetricsBuffer, 'finish', finish)
    pixels, images = DualOutputBuffer('c').finish('cpu')
    assert pixels[-1]['Condition'] == images[-1]['Condition'] == 'dual_output'
    assert pixels[-1]['AUROC'] == 80. and images[-1]['AUROC'] == 90.


def test_changed_reference_identities_block_history_parity(tmp_path):
    for name, identity in [('current', 'new'), ('old', 'original')]:
        folder = tmp_path/name
        folder.mkdir()
        (folder/'references.csv').write_text(f'Identity\n{identity}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='identities'):
        check_history(tmp_path/'current', tmp_path/'old')
