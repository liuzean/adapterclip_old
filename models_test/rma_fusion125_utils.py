"""Experiment L: isolated 12.5% training-fusion control; no shared-file patches.

Schedulers import only standard-library code. Existing new-D/H implementations
are read/reused, never edited or monkey-patched. No CRA is created.
"""
from pathlib import Path

from fusion_ablation_utils import read_csv, read_json
from h_inference import WEIGHT_SHA, recorded_inputs
from rma_align_ablation_utils import DATASETS, metric_index, verified_history
from rma_utils import FIELDS, digest

ROOT=Path(__file__).resolve().parents[1]
TRAIN_FRACTION=.125
LOCAL_CONDITIONS=('L125','L250')


def code_hashes():
    from rma_align_ablation import code_hashes as shared_hashes
    names=('rma_fusion125_utils.py','train_rma_fusion125.py','test_rma_fusion125.py','rma_fusion125_pipeline.py',
           'new_d_control_utils.py')
    return dict(shared_hashes(),**{str(ROOT/'models_test'/n):digest(ROOT/'models_test'/n) for n in names})


def verify_plan(plan):
    if plan['code_sha256']!=code_hashes():raise ValueError('L/shared implementation changed during this run.')
    for p,value in plan['input_sha256'].items():
        if digest(p)!=value:raise ValueError('L input changed: '+p)


def settings(matched):
    """Pin every actual new-D setting; only the fusion-map coefficient changes."""
    from rma_scratch_common import parse_training
    from new_d_control_utils import check_training_settings
    args=parse_training('new',[])
    args.matched_new_d_run=Path(matched).resolve()
    check_training_settings(args)
    manifest=read_json(args.matched_new_d_run/'model_manifest.json')
    for key in ('hidden_dim','top_k','match_chunk','matching'):
        if getattr(args,key)!=manifest['train_options'][key]:raise ValueError('RMA setting differs: '+key)
    state=read_json(args.matched_new_d_run/'run_status.json')
    if state['best_epoch']!=14 or digest(args.matched_new_d_run/'best.pth')!=WEIGHT_SHA:
        raise ValueError('Requires original new-D source-selected epoch14 run.')
    # best.pth and epoch_14.pth may have different torch ZIP container bytes;
    # selected epoch14 is established by its completed source-selection record.
    return args


def training_plan(matched):
    args=settings(matched);old=args.matched_new_d_run
    paths=[args.pretrained_model,args.baseline_checkpoint,args.train_data_path/'meta.json']
    paths += [old/n for n in ('model_manifest.json','run_status.json','source_split.json','training_queries.json',
                            'training_reference_pool.json','training_episodes.csv','training_history.csv','best.pth','epoch_14.pth')]
    return dict(experiment='L',seed=10,epochs=15,train_local_fraction=.125,deploy_local_fraction=.125,
        matched_new_d_run=str(old),code_sha256=code_hashes(),input_sha256={str(p):digest(p) for p in paths},
        selection='Same original new-D source validation: full four-branch25% output, six metrics / three shots; earliest maximum.',
        primary='new epoch14 versus H epoch14; same H image formula and12.5% localization',
        secondary='new source-selected best versus historical new-D source-selected best14; report epoch explicitly',
        only_training_change='Fusion-supervision map: .875*mean(VA,TA,PQA)+.125*RMA; independent branch losses unchanged.')


def training_fusion(pixels, alignment, fraction=TRAIN_FRACTION):
    """Differentiable PRE-smoothing fusion, [B,H,W]; not a branch-loss weight.

    Mirrors rma_joint_utils.joint_forward harmonic arithmetic; only the mixture
    changes. PQA layer probabilities are averaged before branch averaging.
    """
    import torch
    from adaptcliplib import fusion_fun
    if fraction not in (.125,.25) or set(pixels)!={'VA','TA','PQA','RMA'}:
        raise ValueError('Four branches and declared12.5/25% fraction required.')
    branches={n:torch.stack(v).mean(0)[:,1] for n,v in pixels.items()}
    if fraction==.25:
        mixed=torch.stack(list(branches.values())).mean(0)
    else:
        mixed=.875*torch.stack([branches[n] for n in ('VA','TA','PQA')]).mean(0)+.125*branches['RMA']
    align=fusion_fun(alignment,fusion_type='harmonic_mean')[:,0].detach()
    numerator=2*mixed*align;denominator=mixed+align
    if ((denominator==0)&(numerator!=0)).any():raise ValueError('Singular training fusion.')
    result=numerator/torch.where(denominator==0,torch.ones_like(denominator),denominator)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite training fusion.')
    return result


def joint_forward125(model, adapters, rma, features, patches, pf, pp):
    """Same new-D branch calls/order; only the final supervised mixture changes.

    Necessary small forward section from rma_joint_utils.joint_forward, kept
    here so an active CRA/H process never sees a shared-file edit.
    """
    from rma_joint_utils import encode_learned_text
    textual,visual,pq=adapters
    text=encode_learned_text(model,textual)
    va_g,va_p=visual(features,patches,textual.static_text_features)
    ta_g,ta_p=textual.compute_global_local_score(features,patches,text)
    pq_g,pq_p,align=pq(features,patches,pf,pp)
    if len(pq_g)!=4 or len(pq_p)!=4 or len(align)!=4:raise ValueError('Requires four PQA layers.')
    rg,rp=rma(patches,pp)
    globals_={'VA':[va_g],'TA':[ta_g],'PQA':pq_g,'RMA':[rg]}
    pixels={'VA':[va_p],'TA':[ta_p],'PQA':pq_p,'RMA':[rp.softmax(1)]}
    return globals_,pixels,training_fusion(pixels,align)


def target_inputs(reference, dataset, shot):
    """Verify old H/A and25% localization donors, query/reference identity."""
    inputs=recorded_inputs(reference,dataset,shot);history,_=verified_history(inputs)
    old=ROOT/'results/rma_output_ablation/six_datasets_seed10'/dataset/f'{shot}shot_seed10'
    if read_json(old/'run_status.json').get('complete') is not True:raise ValueError('Old25% round incomplete.')
    cfg=read_json(old/'experiment_config.json')
    if cfg['checkpoint_sha256']!=WEIGHT_SHA or cfg['weight_epoch']!=14 or cfg['seed']!=10:
        raise ValueError('Old25% weight/epoch/seed differs.')
    for n in ('references.csv','sample_manifest.csv'):
        if read_csv(old/n)!=read_csv(inputs['history']/n):raise ValueError('Old25% query/reference ledger differs.')
    rows=read_csv(old/'metrics_combined.csv')
    lookup=metric_index(rows,('baseline','G0_L0','G1_L0','G0_L1','G1_L1'),inputs['classes'])
    hist=metric_index(history,('H','baseline'),inputs['classes']);donors=[]
    for c in (*inputs['classes'],'Mean'):
        # This old25% H-score combination is historical metric composition,
        # not a claim of newly running full old-model25% pixel inference.
        donors.append(dict(Category=c,Condition='old_H_L250',**{f:hist[c,'H'][f] if f.startswith('I-') else lookup[c,'G0_L1'][f] for f in FIELDS}))
    return inputs,history+donors,old


def target_plan(reference):
    jobs={};paths=[]
    for d in DATASETS:
        for s in (1,2,4):
            inputs,_,old=target_inputs(reference,d,s)
            jobs[f'{d}/{s}']=dict(queries=len(inputs['manifest']),classes=inputs['classes'])
            paths += [inputs['meta'],Path(inputs['config']['pretrained_model'])]
            paths += [inputs['history']/n for n in ('experiment_config.json','run_status.json','input_audit.json',
                                                   'references.csv','sample_manifest.csv','metrics_combined.csv','image_scores.csv')]
            paths += [old/n for n in ('experiment_config.json','run_status.json','references.csv','sample_manifest.csv','metrics_combined.csv')]
    return dict(jobs=jobs,input_sha256={str(p):digest(p) for p in dict.fromkeys(paths)})


def checked_training(folder):
    """No torch: pin completed training, same-epoch14 and independent source-best."""
    folder=Path(folder).resolve();folder.relative_to((ROOT/'results/train_rma_fusion125').resolve())
    plan=read_json(folder/'campaign_config.json');verify_plan(plan)
    state=read_json(folder/'run_status.json');manifest=read_json(folder/'model_manifest.json')
    if (state.get('complete') is not True or state.get('epochs')!=15 or manifest.get('train_local_fraction')!=.125
        or state.get('matched_episodes_passed') is not True or state.get('frozen_integrity_passed') is not True
        or state.get('four_adapter_gradient_checks_passed') is not True):raise ValueError('L training not complete/verified.')
    for name,value in state['output_sha256'].items():
        if Path(name).name!=name or digest(folder/name)!=value:raise ValueError('L training output changed: '+name)
    history=read_csv(folder/'training_history.csv')
    if [int(r['epoch']) for r in history]!=list(range(1,16)):raise ValueError('Incomplete15 epoch history.')
    validation=read_csv(folder/'validation_metrics.csv')
    for row in history:
        means=[v for v in validation if int(v['epoch'])==int(row['epoch']) and v['Category']=='Mean']
        if len(means)!=3 or {int(v['shot']) for v in means}!={1,2,4}:
            raise ValueError('Incomplete source validation metrics.')
        score=sum(float(v[f]) for v in means for f in FIELDS)/18
        if abs(score-float(row['validation_score']))>1e-9:
            raise ValueError('Source selection score differs from source six metrics.')
    best=max(history,key=lambda r:float(r['validation_score']))
    if int(best['epoch'])!=state['best_epoch'] or abs(float(best['validation_score'])-state['best_score'])>1e-9:
        raise ValueError('Best weight not selected from source validation.')
    return plan,state,dict(epoch14=dict(path=str(folder/'epoch_14.pth'),epoch=14,sha256=digest(folder/'epoch_14.pth')),
                           best=dict(path=str(folder/'best.pth'),epoch=state['best_epoch'],sha256=digest(folder/'best.pth')))


def predict(state, images, classes, paths):
    """One encoder/RMA pass; H scoring stays25%, localization125/250 only."""
    from fusion_ablation import infer_batch
    from h_inference import select_score
    from rma_fixed_fusion_utils import gated_output
    from rma_output_ablation_utils import route_outputs
    from test import prompt_association
    features,patches=state.model.encode_image(images.to(state.device),state.config.features_list,DPAM_layer=20)
    pf,pp=prompt_association(*state.memory,classes)
    shared=(features,pf,*patches,*pp);versions=[v._version for v in shared]
    old,maps,scores,_,_,native,extra=infer_batch(state.config,features,patches,pf,pp,*state.adapters,state.text,paths)
    g,local=state.rma(patches,pp)
    raw,routed,scored,_=route_outputs(old,maps,scores,native,extra,g,local,state.config.sigma)
    _,pixel,_,_=gated_output(old,native,extra,local,None,.125,state.config.sigma,
                            (raw['G0_L1'],routed['G0_L1'],scored['G0_L1']))
    score=select_score(scored['G0_L1'],native.detach().cpu(),'H')
    if versions!=[v._version for v in shared]:raise ValueError('Mutated shared query/reference features.')
    return dict(L125=pixel,L250=routed['G0_L1']),{n:score for n in LOCAL_CONDITIONS},dict(SE=scored['G0_L1'],G0=native.detach().cpu())


def deltas(current, history, classes):
    index=metric_index(current,tuple(sorted({r['Condition'] for r in current})),classes)
    previous=metric_index(history,('H','baseline','old_H_L250'),classes)
    records=[]
    for (category,condition),row in index.items():
        for comparator in ('H','baseline','old_H_L250'):
            change={f:float(row[f])-float(previous[category,comparator][f]) for f in FIELDS}
            records.append(dict(Category=category,Condition=condition,Comparator=comparator,**change,
                                ImageNet=sum(change[f] for f in FIELDS[:3]),PixelNet=sum(change[f] for f in FIELDS[3:]),Net=sum(change.values())))
    return records


def retention(rows):
    from rma_matching_control_utils import retention as assess
    records=[];overall={}
    for condition in sorted({r['Condition'] for r in rows}):
        part=[r for r in rows if r['Condition']==condition]
        report,summary=assess(part,comparators=('H','baseline','old_H_L250'))
        records.extend(dict(Condition=condition,**r) for r in report);overall[condition]=summary
    return records,overall
