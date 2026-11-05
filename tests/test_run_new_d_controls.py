"""Stage/checkpoint/progress contracts, authored for explicit future test runs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from run_new_d_controls import STAGES, progress_text, selected_checkpoint


def training_fixture(tmp_path,complete=True,outside=False,format_='adaptclip-three-scratch-v1'):
    folder=tmp_path/'training_run02'
    folder.mkdir()
    checkpoint=tmp_path/'best.pth' if outside else folder/'best.pth'
    checkpoint.touch()
    (folder/'run_status.json').write_text(json.dumps(dict(complete=complete,flow='new',
        three_adapter_gradient_checks_passed=True,selected_checkpoint=str(checkpoint))),encoding='utf-8')
    (folder/'model_manifest.json').write_text(json.dumps(dict(format=format_,
        initialization='CLIP pretrained; three adapters fresh; no RMA')),encoding='utf-8')
    return folder,checkpoint


def test_sequence_order():
    assert [row[2] for row in STAGES]==['new_d_without_rma.py','train_new_d_three.py','test_new_d_three.py']


def test_uses_exact_reported_training_checkpoint_in_numbered_run(tmp_path):
    folder,checkpoint=training_fixture(tmp_path)
    assert selected_checkpoint(folder)==checkpoint.resolve()


@pytest.mark.parametrize('options',[dict(complete=False),dict(outside=True),dict(format_='adaptclip-rma-scratch-v1')])
def test_rejects_incomplete_external_or_four_adapter_checkpoint(tmp_path,options):
    folder,_=training_fixture(tmp_path,**options)
    with pytest.raises(ValueError,match='Expected this completed'):
        selected_checkpoint(folder)


def test_failure_summary_preserves_epoch_and_test_condition():
    training=progress_text('three_train','三适配器训练',[],Path('output'),
        dict(session='train_run02',last_recorded_epoch=7,total_epochs=15),KeyboardInterrupt())
    assert '7/15' in training and 'three_train' in training and 'Ctrl+C' in training
    testing=progress_text('three_test','三适配器测试',['without_rma','three_train'],Path('output'),
        dict(session='test_run02',completed_rounds=10,total_rounds=18,current=['SDD',2]))
    assert '10/18' in testing and 'SDD' in testing and 'three_test' in testing
