"""Epoch recovery contracts; CPU sampling fixtures, no CLIP/model inference."""
import copy
import os
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_control_resume import replay_sampling,capture_rng,restore_rng,check_legacy_code,LEGACY_TRAIN_SHA


class Metadata(torch.utils.data.Dataset):
    def __init__(self):
        self.root='example'
        self.data_all=[dict(img_path=f'q{i}.png',cls_name='c') for i in range(8)]
    def __len__(self):return len(self.data_all)
    def __getitem__(self,i):
        r=self.data_all[i];return dict(img_path=os.path.join(self.root,r['img_path']),cls_name=r['cls_name'])


def test_metadata_replay_restores_same_sampler_support_state_and_next_queries():
    data=Metadata();loader=torch.utils.data.DataLoader(data,batch_size=2,shuffle=True,num_workers=0)
    val=torch.utils.data.DataLoader(data,batch_size=2,shuffle=False,num_workers=0)
    pool={'c':[dict(img_path=f'r{i}.png') for i in range(6)]}
    torch.manual_seed(10);initial=torch.get_rng_state().clone();rng=random.Random(10)
    rows=[];steps=0
    for epoch in (1,2):
        for step,items in enumerate(loader):
            shot=(1,2,4)[steps%3]
            for path,category in zip(items['img_path'],items['cls_name']):
                chosen=rng.sample(pool[category],shot)
                rows.append(dict(epoch=str(epoch),step=str(step+1),query=path,category=category,shot=str(shot),
                                 references=' | '.join(r['img_path'] for r in chosen)))
            steps+=1
        for _ in (1,2,4):list(val)
    expected_cpu=torch.get_rng_state().clone();expected_support=rng.getstate()
    next_expected=[x['img_path'] for x in loader]
    torch.set_rng_state(initial);rng=random.Random(10)
    audit=replay_sampling(loader,val,pool,rows,2,[1,2,4],rng)
    assert audit['episodes']==16 and audit['model_forward_backward_replayed'] is False
    assert torch.equal(expected_cpu,torch.get_rng_state()) and rng.getstate()==expected_support
    assert next_expected==[x['img_path'] for x in loader]
    torch.set_rng_state(initial);broken=copy.deepcopy(rows);broken[0]['references']='changed'
    with pytest.raises(ValueError,match='replay mismatch'):
        replay_sampling(loader,val,pool,broken,2,[1,2,4],random.Random(10))


def test_saved_rng_roundtrip_weights_only_safe(monkeypatch,tmp_path):
    import numpy as np
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda _:None)
    rng=random.Random(10);state=capture_rng(rng)
    file=tmp_path/'state.pth';torch.save(dict(resume_rng=state),file)
    expected=(random.random(),np.random.rand(),torch.rand(3),rng.random())
    restored=torch.load(file,map_location='cpu',weights_only=True)['resume_rng']
    restore_rng(restored,rng)
    assert random.random()==expected[0] and np.random.rand()==expected[1]
    assert torch.equal(torch.rand(3),expected[2]) and rng.random()==expected[3]


def test_legacy_guard_accepts_unchanged_numerical_code_and_rejects_unknown_version():
    check_legacy_code(dict(source_train_sha=LEGACY_TRAIN_SHA))
    with pytest.raises(ValueError,match='Unknown legacy'):
        check_legacy_code(dict(source_train_sha='unknown'))
