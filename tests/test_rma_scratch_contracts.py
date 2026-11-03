"""Scratch initialization/profile/checkpoint contracts; not automatically run."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_scratch_common import FORMAT, checkpoint_selection, initialize_fresh, parse_training, training_losses, validate_scratch
from rma_utils import MetricsBuffer


def test_original_profile_matches_original_training_settings():
    args = parse_training('original', [])
    assert (args.batch_size,args.accumulation,args.optimizer,args.learning_rate)==(8,1,'adam',.001)
    assert args.betas==[.5,.999] and args.weight_decay==0 and args.train_reference_shots==[1]
    assert args.fusion_weight==args.gradient_clip==0 and args.num_workers==4


def test_new_profile_and_overrides_are_explicit():
    args = parse_training('new', ['--batch_size','4'])
    assert args.overrides==['batch_size'] and args.batch_size==4
    assert args.optimizer=='adamw' and args.accumulation==4 and args.train_reference_shots==[1,2,4]
    assert args.fusion_weight==args.gradient_clip==1


def test_wrong_flow_and_warm_start_checkpoints_rejected():
    with pytest.raises(ValueError, match='FROM-SCRATCH'):
        validate_scratch({'format':'adaptclip-rma-joint-v1'},'base','clip',{},'original')
    with pytest.raises(ValueError, match='FROM-SCRATCH'):
        validate_scratch(dict(format=FORMAT, flow='new'),'base','clip',{},'original')


def test_original_branch_loss_never_uses_fusion_supervision():
    # NaN fused map must have no influence when original branch-only training is used.
    globals_ = {n:[torch.randn(2,2,requires_grad=True)] for n in ('VA','TA','PQA','RMA')}
    pixels = {n:[torch.softmax(torch.randn(2,2,2,2,requires_grad=True),1)] for n in globals_}
    def focal(pred,target):
        return (pred[:,1]-target).square().mean()
    def dice(pred,target):
        return (pred-target).square().mean()
    loss, parts = training_losses('original', globals_,pixels,torch.full((2,2,2),float('nan')),
                                 torch.tensor([0,1]),torch.zeros(2,2,2),focal,dice,0.)
    assert torch.isfinite(loss) and parts['fusion_loss']==0
    loss.backward()
    assert all(v[0].grad is not None for v in globals_.values())


def test_single_candidate_validation_does_not_create_baseline_metrics():
    buffer = MetricsBuffer('category', conditions=('scratch',))
    assert set(buffer.maps)==set(buffer.scores)=={'scratch'}


def test_original_selection_uses_both_losses_not_last_epoch_or_validation():
    history = [dict(global_loss=2., local_loss=3., validation_score=99.),
               dict(global_loss=1., local_loss=2., validation_score=80.),
               dict(global_loss=.5, local_loss=3., validation_score=100.)]
    best, epoch = float('inf'), None
    for i, row in enumerate(history, 1):
        value, improved = checkpoint_selection('original', row, best)
        if improved:
            best, epoch = value, i
    assert epoch == 2 and best == 3.
    assert checkpoint_selection('original', history[1], best) == (3., False)


def test_new_selection_still_maximizes_source_validation():
    assert checkpoint_selection('new', dict(validation_score=91.,global_loss=8.,local_loss=9.),90.) == (91.,True)


def test_nonfinite_selection_is_rejected():
    with pytest.raises(ValueError,match='finite'):
        checkpoint_selection('original',dict(global_loss=float('nan'),local_loss=1.),float('inf'))


@pytest.mark.parametrize('flow',['original','new'])
def test_fresh_initializer_adapts_path_for_original_string_interface(monkeypatch,flow):
    import train as original_train
    args = parse_training(flow,[])
    assert isinstance(args.pretrained_model,Path)
    class StopBeforeLoading(RuntimeError):
        pass
    def resolver(value):
        assert isinstance(value,str)
        assert value == str(args.pretrained_model)
        raise StopBeforeLoading('checked path type; do not load weights')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(original_train,'resolve_pretrained_model_type',resolver)
    with pytest.raises(StopBeforeLoading):
        initialize_fresh(args,None)
