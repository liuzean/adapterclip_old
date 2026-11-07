"""Sequence contracts; authored for explicit pytest, not automatically run."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
import run_test_h_sequence as entry


def test_H_failure_prevents_E(monkeypatch,tmp_path):
    monkeypatch.setattr(entry,'new_directory',lambda *args:tmp_path)
    called=[]
    def fail(command):
        called.append(command)
        raise RuntimeError('simulated H failure')
    monkeypatch.setattr(entry,'run_child',fail)
    monkeypatch.setattr(entry,'cleanup',lambda:None)
    with pytest.raises(RuntimeError,match='simulated H'):
        entry.main([])
    assert len(called)==1 and '--mode' not in called[0]
    state=entry.read_json(tmp_path/'run_status.json')
    assert state['current_mode']=='H' and not state['completed_modes'] and not state['complete']


def test_only_successful_H_enters_E_and_preserves_order(monkeypatch,tmp_path):
    monkeypatch.setattr(entry,'new_directory',lambda *args:tmp_path)
    called=[];checked=[]
    monkeypatch.setattr(entry,'run_child',lambda command:called.append(command))
    monkeypatch.setattr(entry,'cleanup',lambda:None)
    monkeypatch.setattr(entry,'progress',lambda receipt,*args:dict(session=str(tmp_path/receipt.stem),child_status={}))
    monkeypatch.setattr(entry,'require_complete',lambda mode,report:checked.append(mode))
    entry.main([])
    assert len(called)==2 and '--mode' not in called[0]
    assert called[1][called[1].index('--mode')+1]=='E' and checked==['H','E']
    assert entry.read_json(tmp_path/'run_status.json')['complete']


def test_wrong_mode_or_incomplete_coverage_is_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(entry,'read_json',lambda path:dict(options=dict(mode='E',datasets=list(entry.DATASETS),shots=[1,2,4],verify_history=True)))
    with pytest.raises(ValueError,match='Wrong mode'):
        entry.require_complete('H',dict(session=str(tmp_path),child_status=dict(complete=True,completed_jobs=[],job_results={})))
    with pytest.raises(ValueError,match='without completing'):
        entry.require_complete('H',dict(child_status=dict(complete=False)))


def test_complete18_rounds_are_checked_before_next_stage(tmp_path):
    entry.write_json(tmp_path/'campaign_config.json',dict(options=dict(mode='H',datasets=list(entry.DATASETS),shots=[1,2,4],verify_history=True)))
    jobs={}
    for dataset in entry.DATASETS:
        for shot in (1,2,4):
            folder=tmp_path/dataset/str(shot);folder.mkdir(parents=True)
            (folder/'metrics.csv').write_text('completed metrics')
            entry.write_json(folder/'run_status.json',dict(complete=True,frozen_integrity_passed=True,history_parity_passed=True,
                output_sha256={'metrics.csv':entry.digest(folder/'metrics.csv')}))
            jobs[f'{dataset}/{shot}']=str(folder)
    report=dict(session=str(tmp_path),child_status=dict(complete=True,completed_jobs=list(jobs),job_results=jobs))
    entry.require_complete('H',report)
    (Path(next(iter(jobs.values())))/'metrics.csv').write_text('damaged')
    with pytest.raises(ValueError,match='output changed'):
        entry.require_complete('H',report)


def test_status_never_launches_a_child(monkeypatch,tmp_path,capsys):
    entry.write_json(tmp_path/'run_status.json',dict(complete=False,current_mode='H',completed_modes=[]))
    def forbidden(*args):raise AssertionError('Status launched work')
    monkeypatch.setattr(entry,'run_child',forbidden)
    entry.main(['--status',str(tmp_path)])
    assert '不启动或恢复' in capsys.readouterr().out
