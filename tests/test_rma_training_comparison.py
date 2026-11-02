"""Offline comparison regression contracts; run only when explicitly requested."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models_test'))
from rma_utils import compare_training_runs


def comparison_pair(tmp_path, new_seed=10):
    """Minimal legacy/new manifests: seed belongs at manifest top level."""
    roots = []
    for name, seed in [('old', 10), ('new', new_seed)]:
        root = tmp_path / name
        root.mkdir()
        roots.append(root)
        manifest = dict(seed=seed, base_sha256='base', pretrained_sha256='clip',
                        source_split_sha256='split', model_spec={}, train_options=dict(
                            validation_fraction=.2, fusion_weight=1., matching='learned', hidden_dim=128, top_k=8))
        for filename, payload in {
            'model_manifest.json': manifest,
            'run_status.json': dict(complete=True),
            'campaign_config.json': dict(checkpoint=str(root/'best.pth'), datasets=['Visa'], shots=[1]),
        }.items():
            (root/filename).write_text(json.dumps(payload), encoding='utf-8')
        round_dir = root/'Visa'/'1shot_seed10'
        round_dir.mkdir(parents=True)
        keys = ('seed k_shots dataset test_data_path pretrained_model checkpoint_path image_size '
                'features_list sigma batch_size n_ctx vl_reduction pq_mid_dim pq_context '
                'fusion_type visual_learner textual_learner pq_learner disable_rma').split()
        (round_dir/'experiment_config.json').write_text(json.dumps(dict.fromkeys(keys, 1)), encoding='utf-8')
        (round_dir/'run_status.json').write_text(json.dumps(dict(complete=True, baseline_parity_passed=True)), encoding='utf-8')
        for filename in ('references.csv', 'sample_manifest.csv'):
            (round_dir/filename).write_text('Identity\nsame\n', encoding='utf-8')
        (round_dir/'metrics_combined.csv').write_text(
            'Category,Condition,I-AUROC,I-AP,I-F1max,P-AUROC,P-AP,P-F1max\n'
            'Mean,baseline,90,90,90,90,90,90\n'
            f'Mean,rma,{91 if name == "new" else 90},90,90,90,90,90\n', encoding='utf-8')
    return roots


def test_legacy_top_level_seed_is_supported(tmp_path):
    old, new = comparison_pair(tmp_path)
    output = tmp_path/'comparison'
    compare_training_runs(old, new, output)
    assert (output/'new_minus_old_by_dataset.csv').is_file()
    assert 'False' in (output/'new_minus_old_retention.csv').read_text(encoding='utf-8-sig')


def test_different_top_level_seeds_are_rejected(tmp_path):
    old, new = comparison_pair(tmp_path, new_seed=20)
    with pytest.raises(ValueError, match='seed'):
        compare_training_runs(old, new, tmp_path/'comparison')
    assert not (tmp_path/'comparison').exists()
