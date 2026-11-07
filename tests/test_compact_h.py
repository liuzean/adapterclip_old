"""Compact H/E input and output contracts; authored, not auto-executed."""
import json
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
import test_h as entry
from h_inference import WEIGHT_SHA,recorded_inputs,select_score
from fusion_ablation_utils import write_csv
from rma_utils import digest


def test_default_and_E_switch_are_cwd_independent(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    args=entry.parse_args([])
    assert args.mode=='H' and args.shots==[1,2,4] and len(args.datasets)==6 and args.verify_history
    assert entry.parse_args(['--mode','E']).mode=='E'
    assert args.checkpoint==entry.ROOT/'results/train_rma_scratch_new/mvtec_seed10/best.pth'


def test_score_selection_preserves_E_and_fixed_H_formula():
    e=torch.tensor([.2,.8]);g=torch.tensor([.8,.4])
    assert select_score(e,g,'E') is e
    assert torch.allclose(select_score(e,g,'H'),torch.tensor([.35,.7]))
    with pytest.raises(ValueError):select_score(e,g,'A')


def test_changed_recorded_reference_cannot_silently_be_used(tmp_path):
    data=tmp_path/'data';data.mkdir();folder=tmp_path/'reference/SDD/2shot_seed10';folder.mkdir(parents=True)
    for name in ('r1.png','r2.png','q0.png','q1.png','mask.png'):(data/name).write_bytes(b'fixture')
    rows=[dict(img_path=f'r{i}.png',cls_name='c',anomaly=0) for i in (1,2)]
    queries=[dict(img_path='q0.png',cls_name='c',anomaly=0,mask_path=''),dict(img_path='q1.png',cls_name='c',anomaly=1,mask_path='mask.png')]
    (data/'meta.json').write_text(json.dumps(dict(train={'c':rows},test={'c':queries})))
    (folder/'input_audit.json').write_text(json.dumps(dict(metadata_sha256=digest(data/'meta.json'))))
    config=dict(checkpoint_sha256=WEIGHT_SHA,weight_epoch=14,seed=10,k_shots=2,image_size=518,sigma=4,
                features_list=[6,12,18,24],visual_learner=True,textual_learner=True,pq_learner=True,pq_context=True,test_data_path=str(data))
    (folder/'experiment_config.json').write_text(json.dumps(config))
    refs=[{'Category':'c','Image Path':str(data/row['img_path'])} for row in rows]
    write_csv(folder/'references.csv',list(refs[0]),refs)
    manifest=[{'Category':'c','Image Path':str(data/r['img_path']),'Sample Type':'Anomaly' if r['anomaly'] else 'Normal'} for r in queries]
    write_csv(folder/'sample_manifest.csv',list(manifest[0]),manifest)
    hashes={name:digest(folder/name) for name in ('references.csv','sample_manifest.csv','input_audit.json')}
    (folder/'run_status.json').write_text(json.dumps(dict(complete=True,H_six_metric_parity_passed=True,H_pixel_identity_passed=True,H_output_sha256=hashes)))
    assert [r['img_path'] for r in recorded_inputs(tmp_path/'reference','SDD',2)['references']]==['r1.png','r2.png']
    write_csv(folder/'references.csv',list(refs[0]),list(reversed(refs)))
    with pytest.raises(ValueError,match='ledger changed'):recorded_inputs(tmp_path/'reference','SDD',2)


def test_status_is_read_only(monkeypatch,tmp_path,capsys):
    entry.write_json(tmp_path/'run_status.json',dict(complete=False,current='MPDD/2'))
    def forbidden(*args):raise AssertionError('Status launched inference')
    monkeypatch.setattr(entry,'run_child',forbidden)
    entry.main(['--status',str(tmp_path)])
    assert 'MPDD/2' in capsys.readouterr().out
