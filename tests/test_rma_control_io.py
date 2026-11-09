"""Windows sharing-failure regression contracts. No model construction or GPU."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
import rma_control_io as io


def denied():
    error=PermissionError(13,'simulated sharing conflict')
    error.winerror=5
    return error


def test_transient_replace_failure_retries_without_truncating_old_file(monkeypatch,tmp_path):
    path=tmp_path/'progress.json';path.write_text('{"epoch":12}',encoding='utf-8')
    original=io.os.replace;attempts=[]
    def replace(source,target):
        attempts.append(str(source))
        if len(attempts)<3:
            assert json.loads(path.read_text())==dict(epoch=12)
            raise denied()
        original(source,target)
    monkeypatch.setattr(io.os,'replace',replace);monkeypatch.setattr(io.time,'sleep',lambda _:None)
    assert io.write_json(path,dict(epoch=13)) is True
    assert len(attempts)==3 and json.loads(path.read_text())==dict(epoch=13)
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_progress_failure_is_nonfatal_and_preserves_previous_file(monkeypatch,tmp_path,capsys):
    path=tmp_path/'progress.json';path.write_text('{"epoch":12}',encoding='utf-8')
    def fail(*_):raise denied()
    monkeypatch.setattr(io.os,'replace',fail);monkeypatch.setattr(io.time,'sleep',lambda _:None)
    assert io.write_progress(path,dict(epoch=13)) is False
    assert io.write_progress(path,dict(epoch=14)) is False
    assert json.loads(path.read_text())==dict(epoch=12)
    assert capsys.readouterr().err.count('继续运行')==1
    assert not list(tmp_path.glob('*.tmp'))


def test_critical_completion_write_still_raises_after_retries(monkeypatch,tmp_path):
    path=tmp_path/'run_status.json';path.write_text('{"complete":false}',encoding='utf-8')
    def fail(*_):raise denied()
    monkeypatch.setattr(io.os,'replace',fail);monkeypatch.setattr(io.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):io.write_json(path,dict(complete=True))
    assert json.loads(path.read_text())==dict(complete=False)
    assert not list(tmp_path.glob('*.tmp'))


def test_unique_temporary_files_and_invalid_data_never_report_success(monkeypatch,tmp_path):
    original=io.os.replace;names=[]
    def capture(source,target):names.append(str(source));original(source,target)
    monkeypatch.setattr(io.os,'replace',capture)
    path=tmp_path/'progress.json'
    io.write_progress(path,dict(epoch=1));io.write_progress(path,dict(epoch=2))
    assert len(set(names))==2
    with pytest.raises(ValueError):io.write_progress(path,dict(epoch=float('nan')))
    assert json.loads(path.read_text())==dict(epoch=2)


def test_training_progress_keeps_latest_in_memory_after_failed_write(monkeypatch,tmp_path):
    import rma_matching_control_utils as control
    calls=[]
    def locked(_):raise PermissionError('reader locked')
    path=tmp_path/'training_progress.json';path.write_text('{}',encoding='utf-8')
    monkeypatch.setattr(control,'read_json',locked)
    def save(path,data,*,critical):
        assert critical is False;calls.append(dict(data));return False
    monkeypatch.setattr(control,'control_json',save)
    assert control.record_training_progress(tmp_path,'training',epoch=13,batch=690) is False
    control.record_training_progress(tmp_path,'source_validation',validation_batch=62)
    assert calls[-1]['epoch']==13 and calls[-1]['batch']==690 and calls[-1]['validation_batch']==62
