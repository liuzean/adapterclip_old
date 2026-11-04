"""Sequence/checkpoint receipt contracts; authored for explicit future test runs."""
import json
import sys
import subprocess
import os
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from run_rma_experiment_sequence import STAGES, selected_checkpoint, resume_stages


def test_training_and_testing_order_is_fixed():
    assert [s[0] for s in STAGES] == ['dual_output','original_train','original_test','new_train','new_test']


def make_training_session(tmp_path, flow='original', complete=True, outside=False):
    session = tmp_path/'training'
    session.mkdir()
    best = tmp_path/'best.pth' if outside else session/'best.pth'
    best.touch()
    (session/'run_status.json').write_text(json.dumps(dict(flow=flow,complete=complete,selected_checkpoint=str(best))),encoding='utf-8')
    (session/'model_manifest.json').write_text(json.dumps(dict(flow=flow,format='adaptclip-rma-scratch-v1')),encoding='utf-8')
    return session,best


def test_uses_checkpoint_from_its_own_completed_training(tmp_path):
    session,best = make_training_session(tmp_path)
    assert selected_checkpoint(session,'original') == best.resolve()


@pytest.mark.parametrize('flow,complete,outside', [('new',True,False),('original',False,False),('original',True,True)])
def test_wrong_flow_incomplete_or_external_checkpoint_stops_pipeline(tmp_path,flow,complete,outside):
    session,_ = make_training_session(tmp_path,flow,complete,outside)
    with pytest.raises(ValueError,match='matching best'):
        selected_checkpoint(session,'original')


def test_scratch_import_bootstraps_project_root_from_another_cwd(tmp_path):
    scripts = Path(__file__).resolve().parents[1]/'models_test'
    code = ('import sys, importlib.util; sys.path.insert(0, '+repr(str(scripts))+'); '
            'import rma_scratch_common; '
            'assert importlib.util.find_spec("adaptcliplib") is not None; '
            'assert "torch" not in sys.modules')
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,env=env,check=True)


def test_resume_rejects_noncontiguous_completed_stages(tmp_path,monkeypatch):
    import run_rma_experiment_sequence as sequence
    monkeypatch.setattr(sequence,'ROOT',tmp_path)
    previous = tmp_path/'results'/'run_rma_experiment_sequence'/'previous'
    previous.mkdir(parents=True)
    (previous/'run_status.json').write_text(json.dumps(dict(completed_stages=['original_train'])),encoding='utf-8')
    with pytest.raises(ValueError,match='contiguous'):
        resume_stages(previous)
