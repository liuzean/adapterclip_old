"""CRA CPU regression contracts; named separately from the test_cra entry point."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from cra_model import ContextResidualAggregation, PQAResidualHook
from cra_utils import compare
from rma_utils import FIELDS, check_frozen, freeze_base


def test_zero_gate_identity_no_mutation_and_gate_gradient():
    torch.manual_seed(10)
    model = ContextResidualAggregation(dim=8)
    x = torch.randn(2, 8, 4, 4); before = x.clone(); version = x._version
    y = model(x, 0)
    assert torch.equal(y, x) and torch.equal(x, before) and x._version == version
    y.square().mean().backward()
    assert model.gates.grad is not None and torch.isfinite(model.gates.grad).all()
    assert model.gates.grad[0].abs() > 0
    assert torch.count_nonzero(model.convs[0].weight.grad) == 0
    assert sum(p.numel() for p in ContextResidualAggregation().parameters()) == 27652


def test_original_PQA_hooks_recover_baseline_and_only_CRA_receives_gradients():
    from adaptcliplib import PQAdapter
    torch.manual_seed(9)
    pq = PQAdapter(56, 14, context=True, input_dim=8, mid_dim=8, layers_num=4).eval().requires_grad_(False)
    frozen = freeze_base([None, pq])
    q = [torch.randn(2, 17, 8) for _ in range(4)]
    p = [torch.randn(2, 1, 17, 8) for _ in range(4)]
    with torch.no_grad(): baseline = pq(None, q, None, p)
    cra = ContextResidualAggregation(dim=8)
    hook = PQAResidualHook(pq, cra)
    try:
        assert not any(id(m) == id(cra) for m in pq.modules())
        with torch.no_grad(): zero = pq(None, q, None, p)
        assert all(torch.equal(a,b) for xs,ys in zip(baseline,zero) for a,b in zip(xs,ys))
        assert hook.calls == 4
        with torch.no_grad():
            cra.gates.fill_(.3)
            for conv in cra.convs: conv.weight.fill_(.2)
        changed = pq(None, q, None, p)
        assert any(not torch.equal(a,b) for a,b in zip(changed[1],baseline[1]))
        assert all(torch.equal(a,b) for a,b in zip(changed[2],baseline[2]))
        loss = sum(v.square().mean() for v in changed[0]) + sum(v[:,1].square().mean() for v in changed[1])
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in cra.parameters())
        check_frozen(frozen)
        with hook.active(False), torch.no_grad(): restored = pq(None, q, None, p)
        assert all(torch.equal(a,b) for xs,ys in zip(baseline,restored) for a,b in zip(xs,ys))
        assert hook.enabled is True
    finally: hook.close()
    assert not any(bn._forward_pre_hooks for bn in pq.sharebn)


def test_hook_context_restores_state_after_exception():
    pq = torch.nn.Module(); pq.context = True
    pq.sharebn = torch.nn.ModuleList([torch.nn.BatchNorm2d(8) for _ in range(4)])
    hook = PQAResidualHook(pq, ContextResidualAggregation(dim=8))
    try:
        with pytest.raises(RuntimeError), hook.active(False):
            raise RuntimeError('simulated interrupted inference')
        assert hook.enabled is True
    finally: hook.close()


def test_CRA_initialization_preserves_parent_CPU_rng():
    from types import SimpleNamespace
    from cra_utils import attach
    pq = torch.nn.Module(); pq.context = True
    pq.sharebn = torch.nn.ModuleList([torch.nn.BatchNorm2d(768) for _ in range(4)])
    state = SimpleNamespace(device='cpu',adapters=(None,None,pq))
    before = torch.get_rng_state().clone()
    cra, hook = attach(state)
    try:
        assert torch.equal(before,torch.get_rng_state()) and torch.count_nonzero(cra.gates)==0
    finally: hook.close()


def test_candidate_uses_all_metrics_and_can_change_image_scores():
    def row(c,n,value): return dict(Category=c,Condition=n,**{f:value for f in FIELDS})
    current = [row(c,n,51 if n=='K_CRA' else 50) for c in ('a','Mean') for n in ('H','K_CRA')]
    previous = [row(c,n,50) for c in ('a','Mean') for n in ('H','baseline')]
    combined, checks, deltas = compare(current,previous,['a'])
    assert len(combined)==6 and all(r['Passed'] for r in checks)
    assert all(r['ImageNet']==3 and r['PixelNet']==3 and r['Net']==6 for r in deltas)
    current[0]['P-AP'] += .1
    assert not all(r['Passed'] for r in compare(current,previous,['a'])[1])


def test_source_best_epoch0_is_allowed_and_changed_selection_is_rejected(monkeypatch,tmp_path):
    import cra_utils
    from fusion_ablation_utils import write_csv
    from rma_control_io import write_json
    from rma_utils import digest
    monkeypatch.setattr(cra_utils,'ROOT',tmp_path)
    monkeypatch.setattr(cra_utils,'code_hashes',lambda:{})
    folder = tmp_path/'results/train_cra/run'; folder.mkdir(parents=True)
    weight = folder/'best.pth'; weight.write_bytes(b'checkpoint identity only; no torch load')
    write_json(folder/'campaign_config.json',dict(code_sha256={},input_sha256={}))
    write_csv(folder/'training_history.csv',['epoch','validation_score'],
        [dict(epoch=e,validation_score=90 if e==0 else 89) for e in range(16)])
    (folder/'source_H_metrics.csv').write_text('source control fixture',encoding='utf-8')
    state = dict(complete=True,completed_epochs=15,best_epoch=0,frozen_integrity_passed=True,
        best_score=90,gradient_checks_passed=True,zero_init_parity_passed=True,selected_checkpoint_sha256=digest(weight),
        training_history_sha256=digest(folder/'training_history.csv'),source_H_sha256=digest(folder/'source_H_metrics.csv'))
    write_json(folder/'run_status.json',state)
    assert cra_utils.checked_checkpoint(weight)[1]['best_epoch']==0
    write_json(folder/'run_status.json',dict(state,best_epoch=15))
    with pytest.raises(ValueError): cra_utils.checked_checkpoint(weight)


def test_nested_progress_only_reports_requested_training_or_test_position(tmp_path):
    from cra_workflow import describe
    from rma_control_io import write_json
    outer=tmp_path/'outer';pipeline=tmp_path/'pipeline';training=tmp_path/'training'
    for path in (outer,pipeline,training): path.mkdir()
    write_json(outer/'run_status.json',dict(current_stage='cra'))
    write_json(outer/'cra_progress.json',dict(session=str(pipeline)))
    write_json(pipeline/'run_status.json',dict(current_stage='train'))
    write_json(pipeline/'train_progress.json',dict(session=str(training)))
    write_json(training/'training_progress.json',dict(epoch=8,batch=41))
    assert describe(outer)=='阶段=cra → 阶段=train → 训练：epoch=8，batch=41'
    write_json(training/'run_status.json',dict(current='DTD/2'))
    (training/'training_progress.json').unlink()
    assert describe(outer)=='阶段=cra → 阶段=train → 测试：数据集=DTD，shot=2'
