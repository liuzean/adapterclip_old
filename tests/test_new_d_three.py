"""Meaningful three-control loss/checkpoint contracts; not auto-executed."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_joint_utils import joint_losses, require_joint_gradients
from rma_scratch_common import THREE_FORMAT, THREE_INITIALIZATION, parse_training, validate_three_scratch


def test_three_control_keeps_new_profile_but_has_distinct_identity():
    args=parse_training('new',[],three_adapter_control=True)
    assert args.three_adapter_control and args.batch_size==2 and args.accumulation==4
    assert args.optimizer=='adamw' and args.learning_rate==1e-4
    assert args.train_reference_shots==[1,2,4] and args.fusion_weight==1


def test_three_branch_loss_retains_fusion_gradients_and_has_no_rma_terms():
    global_logits={n:torch.randn(2,2,requires_grad=True) for n in ('VA','TA','PQA')}
    local_logits={n:torch.randn(2,2,3,3,requires_grad=True) for n in global_logits}
    raw=torch.full((2,3,3),.4,requires_grad=True)
    def focal(pred,target):
        return (pred[:,1]-target).square().mean()
    def dice(pred,target):
        return (pred-target).square().mean()
    loss,parts=joint_losses({n:[v] for n,v in global_logits.items()},
        {n:[v.softmax(1)] for n,v in local_logits.items()},raw,torch.tensor([0,1]),
        torch.zeros(2,3,3),focal,dice,1.)
    loss.backward()
    assert torch.isfinite(loss) and raw.grad is not None and torch.count_nonzero(raw.grad)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in global_logits.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in local_logits.values())
    assert not any('RMA' in key for key in parts)
    assert torch.allclose(loss,parts['global_loss']+parts['local_loss']+parts['fusion_loss'])


def test_three_checkpoint_rejects_four_adapter_payload_even_if_old_three_keys_exist():
    payload=dict(format=THREE_FORMAT,flow='new',initialization=THREE_INITIALIZATION,
                 textual_learner={},visual_learner={},pq_learner={},rma={})
    with pytest.raises(ValueError,match='not new D'):
        validate_three_scratch(payload,'base','clip',{},'new')


def test_three_gradient_guard_does_not_require_a_dummy_fourth_adapter():
    modules=[torch.nn.Linear(2,1) for _ in range(3)]
    for module in modules:
        module(torch.ones(2,2)).sum().backward()
    require_joint_gradients(modules,None)
    modules[0].zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError,match='TA'):
        require_joint_gradients(modules,None)
