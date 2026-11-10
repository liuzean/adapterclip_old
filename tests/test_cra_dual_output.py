"""K dual-output CPU contracts; no automatic model inference or test execution."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from cra_dual_output_utils import DUAL,DualBuffer,compare,route
from rma_utils import FIELDS,MetricsBuffer


def test_route_aliases_final_donors_without_rescoring(monkeypatch):
    maps={'H':torch.full((2,3,3),.1),'K_CRA':torch.full((2,3,3),.9)}
    scores={'H':torch.tensor([.2,.3]),'K_CRA':torch.tensor([.7,.8])}
    def forbidden(*args,**kwargs):raise AssertionError('No map-max recomputation allowed.')
    monkeypatch.setattr(torch.Tensor,'max',forbidden)
    route(maps,scores)
    assert maps[DUAL] is maps['H'] and scores[DUAL] is scores['K_CRA']
    assert maps[DUAL] is not maps['K_CRA'] and scores[DUAL] is not scores['H']


def test_dual_buffer_keeps_two_maps_and_rejects_changed_donor():
    maps={'H':torch.full((1,3,3),.1),'K_CRA':torch.full((1,3,3),.9)}
    scores={'H':torch.tensor([.2]),'K_CRA':torch.tensor([.8])};route(maps,scores)
    buffer=DualBuffer('c');buffer.append(torch.zeros(3,3),0,maps,scores,0)
    assert set(buffer.maps)=={'H','K_CRA'} and len(buffer.maps['H'])==1
    maps[DUAL]=maps['K_CRA']
    with pytest.raises(ValueError):buffer.append(torch.zeros(3,3),0,maps,scores,0)
    route(maps,scores);scores[DUAL]=scores['H']
    with pytest.raises(ValueError):buffer.append(torch.zeros(3,3),0,maps,scores,0)


def test_finish_reuses_the_exact_donor_metric_rows(monkeypatch):
    def finish(self,device):
        return [dict(Category='c',Condition=n,AUROC=v,AP=v,F1max=v) for n,v in [('H',10),('K_CRA',20)]],\
               [dict(Category='c',Condition=n,AUROC=v,AP=v,F1max=v) for n,v in [('H',30),('K_CRA',40)]]
    monkeypatch.setattr(MetricsBuffer,'finish',finish)
    pixels,images=DualBuffer('c').finish('cpu')
    assert pixels[-1]['Condition']==images[-1]['Condition']==DUAL
    assert pixels[-1]['AP']==10 and images[-1]['AP']==40


def test_history_identity_and_offline_checks_are_independent():
    def row(c,n,iv,pv):return dict(Category=c,Condition=n,**{f:iv if f.startswith('I-') else pv for f in FIELDS})
    current=[row(c,n,iv,pv) for c in ('a','Mean') for n,iv,pv in [('H',50,60),('K_CRA',55,58),(DUAL,55,60)]]
    old=[r for r in current if r['Condition']!=DUAL]
    base=[row(c,'baseline',49,59) for c in ('a','Mean')]
    offline={f:55 if f.startswith('I-') else 60 for f in FIELDS}
    checks,identity,off,deltas=compare(current,old,base,['a'],offline)
    assert all(r['Passed'] for r in [*checks,*identity,*off])
    assert {r['Comparator'] for r in deltas}=={'H','K_CRA','baseline'}
    changed=[dict(r,**{'P-AP':59}) if r['Condition']==DUAL else dict(r) for r in current]
    assert not all(r['Passed'] for r in compare(changed,old,base,['a'],offline)[1])
    assert not all(r['Passed'] for r in compare(current,old,base,['a'],dict(offline,**{'I-AP':54}))[2])
