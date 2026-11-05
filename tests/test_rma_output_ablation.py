"""Four-condition computation/alias contracts; authored, not auto-executed."""
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from rma_output_ablation import parse_args
from rma_output_ablation_utils import FactorBuffer, route_outputs
from rma_utils import fuse_rma


def fixture_outputs():
    from adaptcliplib import fusion_fun
    raw=torch.full((2,2,2),.3)
    native=torch.full((2,),.2)
    old={'baseline':raw,'no_align':torch.full_like(raw,.2)}
    maps={'baseline':raw.clone()}
    scores={'baseline':fusion_fun([native,maps['baseline'].flatten(1).max(1).values],fusion_type='harmonic_mean')}
    extra={'ALIGN':torch.full_like(raw,.6)}
    global_logits=torch.tensor([[0.,1.4],[0.,1.4]])
    local_logits=global_logits[:,:,None,None].expand(2,2,2,2).clone()
    return old,maps,scores,native,extra,global_logits,local_logits


def test_default_scope_and_order():
    args=parse_args([])
    assert args.datasets==['Visa','BTAD','MPDD','SDD','DTD','part_glass'] and args.shots==[1,2,4]


def test_existing_all_on_and_all_off_paths_are_preserved():
    old,maps,scores,native,extra,g,l=fixture_outputs()
    raw,output,image,_=route_outputs(old,maps,scores,native,extra,g,l,0)
    expected_raw,expected_map,expected_score=fuse_rma(old,extra,native,g,l,0)
    assert raw['G0_L0'] is old['baseline'] and output['G0_L0'] is maps['baseline']
    assert image['G0_L0'] is scores['baseline']
    assert torch.equal(raw['G1_L1'],expected_raw) and torch.equal(output['G1_L1'],expected_map)
    assert torch.equal(image['G1_L1'],expected_score)


def test_global_switch_keeps_pixel_maps_and_local_switch_can_change_image_score():
    _,maps,scores,parts=route_outputs(*fixture_outputs(),0)
    assert maps['G1_L0'] is maps['G0_L0'] and maps['G0_L1'] is maps['G1_L1']
    assert torch.all(scores['G1_L0']>scores['G0_L0'])
    assert torch.all(scores['G0_L1']>scores['G0_L0'])
    assert torch.all(parts['G1']>parts['G0'])


def test_only_two_candidate_map_copies_are_buffered():
    _,maps,scores,_=route_outputs(*fixture_outputs(),0)
    maps['baseline']=maps['G0_L0'];scores['baseline']=scores['G0_L0']
    buffer=FactorBuffer('class')
    buffer.append(torch.zeros(2,2),0,maps,scores,0)
    assert set(buffer.maps)=={'baseline','G0_L0','G1_L1'}
    assert sum(len(v) for v in buffer.maps.values())==3
    assert set(buffer.extra_scores)=={'G1_L0','G0_L1'}
