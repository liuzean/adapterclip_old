"""Experiment I numerical/identity contracts; authored, not automatically run."""
import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_model import ReferenceMatchingAdapter
from rma_scratch_common import parse_training,checkpoint_selection
from h_inference import load_model,WEIGHT_SHA
from rma_matching_control_utils import output_table,retention
from rma_utils import FIELDS


def models():
    torch.manual_seed(17)
    learned=ReferenceMatchingAdapter(input_dim=8,hidden_dim=8,top_k=3,image_size=4,chunk=2)
    cosine=ReferenceMatchingAdapter(input_dim=8,hidden_dim=8,top_k=3,image_size=4,chunk=2,matching='cosine')
    cosine.load_state_dict(learned.state_dict())
    q=torch.randn(2,5,8);r=torch.randn(2,2,5,8)
    return learned,cosine,q,r


def test_cosine_is_exact_learned_formula_with_zero_QK_term():
    learned,cosine,q,r=models()
    with torch.no_grad():learned.key.weight.zero_();learned.key.bias.zero_()
    a=learned.layer(q,r);b=cosine.layer(q,r)
    assert all(torch.allclose(x,y,atol=1e-6,rtol=1e-5) for x,y in zip(a,b))


def test_cosine_ignores_K_but_keeps_Q_value_relation_gradients():
    _,model,q,r=models();first=model.layer(q,r)
    with torch.no_grad():model.key.weight.fill_(100);model.key.bias.fill_(-200)
    second=model.layer(q,r)
    assert all(torch.equal(x,y) for x,y in zip(first,second))
    (second[0].square().mean()+second[1].square().mean()).backward()
    assert all(p.grad is None and not p.requires_grad for p in model.key.parameters())
    for module in (model.query,model.value,model.relation,model.local_head,model.global_head):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())


def test_same_topk_candidate_set_and_shape():
    learned,cosine,q,r=models()
    # Retrieval uses exactly the same frozen q/r for both, including chunk ties.
    indices=[]
    original=torch.Tensor.topk
    def spy(self,*args,**kwargs):
        result=original(self,*args,**kwargs);indices.append(result.indices.clone());return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor,'topk',spy)
        a=learned([q]*4,[r]*4);half=len(indices);b=cosine([q]*4,[r]*4)
    assert half and len(indices)==2*half
    assert all(torch.equal(x,y) for x,y in zip(indices[:half],indices[half:]))
    assert a[0].shape==b[0].shape==(2,2) and a[1].shape==b[1].shape==(2,2,4,4)


def test_control_is_locked_to_cosine_new_flow_and_H_defaults_stay_strict():
    args=parse_training('new',[],matching_control=True)
    assert args.matching=='cosine' and args.epochs==15 and args.train_reference_shots==[1,2,4]
    with pytest.raises(SystemExit):parse_training('new',['--matching','uniform'],matching_control=True)
    assert parse_training('new',[]).matching=='learned'
    signature=inspect.signature(load_model)
    assert signature.parameters['expected_sha'].default==WEIGHT_SHA
    assert signature.parameters['expected_epoch'].default==14
    assert signature.parameters['expected_matching'].default=='learned'
    assert checkpoint_selection('new',dict(validation_score=90),89)==(90,True)


def test_complete_metrics_and_partial_shots_cannot_claim_retention():
    row=dict(Category='Mean',Condition='I_cosine',**{f:51 for f in FIELDS})
    old=[dict(Category='Mean',Condition=n,**{f:50 for f in FIELDS}) for n in ('H','baseline')]
    table,deltas=output_table([row],old)
    assert len(table)==3 and len(deltas)==2 and all(r['Net']==6 for r in deltas)
    from context_alignment import DATASETS
    records=[dict(Dataset=d,Shot=1,**r) for d in DATASETS for r in deltas]
    decisions,overall=retention(records)
    assert sum(r['Pass'] for r in decisions)==2
    assert all(not r['retain'] and not r['complete'] for r in overall.values())
    reverse=[dict(r,Comparator='I_cosine',**{f:-float(r[f]) for f in (*FIELDS,'ImageNet','PixelNet','Net')})
             for r in records if r['Comparator']=='H']
    _,learned=retention(reverse,comparators=('I_cosine',))
    assert not learned['I_cosine']['retain']
    with pytest.raises(ValueError):output_table([dict(row,Category='other')],old)


def test_stop_description_only_reports_training_epoch_and_batch(tmp_path):
    from fusion_ablation_utils import write_json
    from run_rma_matching_control import describe,inspect_progress,stop_position
    child=tmp_path/'training';child.mkdir()
    write_json(tmp_path/'train_progress.json',dict(session=str(child)))
    write_json(child/'run_status.json',dict(epoch=2,complete=False))
    write_json(child/'training_progress.json',dict(phase='training',epoch=3,completed_epochs=2,
        batch=17,total_batches=690,completed_batches=16))
    state=dict(current_stage='train',completed_stages=[],error='Ctrl+C')
    snapshot=inspect_progress(tmp_path,state);text=describe(tmp_path,state,snapshot)
    assert text=='训练：epoch=3，batch=17'
    assert stop_position(state,snapshot)==dict(stage='train',epoch=3,batch=17)


def test_stop_description_only_reports_dataset_shot_and_survives_broken_worker(tmp_path):
    from fusion_ablation_utils import write_json
    from run_rma_matching_control import describe,inspect_progress
    child=tmp_path/'testing';child.mkdir();worker=child/'round';worker.mkdir()
    write_json(tmp_path/'test_progress.json',dict(session=str(child)))
    write_json(child/'run_status.json',dict(current='SDD/2',current_output=str(worker),completed_jobs=['SDD/1']))
    write_json(worker/'run_status.json',dict(phase='evaluating',processed_queries=24,total_queries=100))
    state=dict(current_stage='test',completed_stages=['train'])
    text=describe(tmp_path,state)
    assert text=='测试：数据集=SDD，shot=2'
    (worker/'run_status.json').write_text('{broken',encoding='utf-8')
    snapshot=inspect_progress(tmp_path,state)
    assert snapshot['read_errors'] and describe(tmp_path,state,snapshot)=='测试：数据集=SDD，shot=2'
