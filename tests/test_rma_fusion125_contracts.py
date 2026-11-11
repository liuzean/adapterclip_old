"""CPU contracts for L; distinct module name from the experiment test entry."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_fusion125_utils import training_fusion


def inputs():
    values={n:torch.tensor([[[[1-v]],[[v]]]],requires_grad=True) for n,v in [('VA',.2),('TA',.4),('PQA',.6),('RMA',.8)]}
    return {n:[v] for n,v in values.items()},[torch.full((1,1,1,1),.5)]


def test_fraction125_matches_manual_pre_align_mix():
    pixels,align=inputs();value=training_fusion(pixels,align)
    mixed=.875*(.2+.4+.6)/3+.125*.8
    assert float(value)==pytest.approx(2*mixed*.5/(mixed+.5))


def test_fraction250_matches_original_equal_branch_mean():
    pixels,align=inputs();value=training_fusion(pixels,align,.25)
    mixed=(.2+.4+.6+.8)/4
    assert float(value)==pytest.approx(2*mixed*.5/(mixed+.5))


def test_fusion_keeps_finite_gradients_for_all_branches():
    pixels,align=inputs();value=training_fusion(pixels,align);value.sum().backward()
    for branch in pixels.values():
        grad=branch[0].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_invalid_fraction_rejected_and_zero_harmonic_finite():
    pixels,align=inputs()
    with pytest.raises(ValueError):training_fusion(pixels,align,.5)
    zero={n:[torch.zeros_like(v[0],requires_grad=True)] for n,v in pixels.items()}
    value=training_fusion(zero,[torch.zeros_like(align[0])]);value.sum().backward()
    assert torch.isfinite(value).all() and float(value)==0


def test_fusion_does_not_modify_branch_probability_tensors():
    pixels,align=inputs();before={n:v[0].detach().clone() for n,v in pixels.items()}
    training_fusion(pixels,align)
    assert all(torch.equal(v[0],before[n]) for n,v in pixels.items())


def test_only_fusion_loss_changes_not_independent_branch_losses():
    from rma_scratch_common import training_losses
    pixels,align=inputs()
    globals_={name:[torch.tensor([[.1,.9]],requires_grad=True)] for name in pixels}
    labels=torch.tensor([1]);masks=torch.ones((1,1,1))
    focal=lambda prediction,target: prediction.square().mean()
    dice=lambda prediction,target: (prediction-target).square().mean()
    parts=[]
    for fraction in (.125,.25):
        _,values=training_losses('new',globals_,pixels,training_fusion(pixels,align,fraction),labels,masks,focal,dice,1.)
        parts.append(values)
    assert torch.equal(parts[0]['global_loss'],parts[1]['global_loss'])
    assert torch.equal(parts[0]['local_loss'],parts[1]['local_loss'])
    assert not torch.equal(parts[0]['fusion_loss'],parts[1]['fusion_loss'])
