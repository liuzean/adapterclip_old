"""Dataset-level recovery contracts; authoring only, no automatic execution."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
import rma_scratch_resume_utils as resume


def fixture(tmp_path):
    datasets=['Visa','BTAD','MPDD','SDD','DTD','part_glass']
    checkpoint=tmp_path/'best.pth'
    checkpoint.write_bytes(b'checkpoint fixture')
    previous=tmp_path/'results/test_rma_scratch_original/old'
    previous.mkdir(parents=True)
    args=SimpleNamespace(checkpoint=checkpoint,flow='original',datasets=datasets,shots=[1,2,4])
    (previous/'campaign_config.json').write_text(json.dumps(dict(flow='original',checkpoint=str(checkpoint),
        checkpoint_sha256=resume.digest(checkpoint),datasets=datasets,shots=[1,2,4],seed=10)),encoding='utf-8')
    jobs=[(d,s) for d in datasets for s in (1,2,4)]
    audit={d:{str(s):{} for s in (1,2,4)} for d in datasets}
    return previous,args,jobs,audit


@pytest.mark.parametrize('failed_shot',[1,2,4])
def test_any_incomplete_DTD_shot_reruns_entire_DTD_and_later_datasets(tmp_path,monkeypatch,failed_shot):
    previous,args,jobs,audit=fixture(tmp_path)
    def recorded_round(directory,flow,checkpoint_hash,round_audit,dataset,shot):
        if dataset=='DTD' and shot==failed_shot:
            raise ValueError('incomplete or zero-filled state')
    monkeypatch.setattr(resume,'audit_round',recorded_round)
    reused,reason=resume.reusable_datasets(previous,args,jobs,audit,tmp_path,'test_rma_scratch_original')
    assert reused==jobs[:12] and reason['dataset']=='DTD'
    assert not any(d in ('DTD','part_glass') for d,s in reused)


def test_changed_checkpoint_cannot_reuse_previous_predictions(tmp_path):
    previous,args,jobs,audit=fixture(tmp_path)
    args.checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='different checkpoint'):
        resume.reusable_datasets(previous,args,jobs,audit,tmp_path,'test_rma_scratch_original')


def test_zero_filled_status_is_not_accepted_as_complete(tmp_path):
    (tmp_path/'run_status.json').write_bytes(b'\x00'*454)
    with pytest.raises(ValueError):
        resume.audit_round(tmp_path,'original','hash',{},'DTD',1)


def test_copy_round_keeps_original_bytes_and_refuses_overwrite(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new'
    source=old/'Visa/1shot_seed10'
    source.mkdir(parents=True)
    (source/'metrics_combined.csv').write_bytes(b'preserved test results')
    resume.copy_round(old,new,'Visa',1)
    assert (source/'metrics_combined.csv').read_bytes()==b'preserved test results'
    assert (new/'Visa/1shot_seed10/metrics_combined.csv').read_bytes()==b'preserved test results'
    with pytest.raises(FileExistsError):
        resume.copy_round(old,new,'Visa',1)
