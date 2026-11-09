"""Experiment I identity/selection/metric checks; no torch in parent processes."""
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fusion_ablation_utils import read_csv, read_json, write_csv, write_json
from h_inference import WEIGHT_SHA
from rma_utils import FIELDS, digest
from rma_control_io import write_json as control_json

INPUT_FILES = ('source_split.json','training_queries.json','training_reference_pool.json','training_episodes.csv')
_PROGRESS_CACHE={}


def record_training_progress(output, phase, **details):
    """Persist last observed work, independently of epoch checkpoint completion."""
    path=Path(output)/'training_progress.json'
    key=str(path.resolve())
    if key not in _PROGRESS_CACHE:
        try:previous=read_json(path) if path.is_file() else {}
        except (OSError,ValueError):previous={}
        _PROGRESS_CACHE[key]=previous if isinstance(previous,dict) else {}
    previous=_PROGRESS_CACHE[key]
    previous.update(phase=phase,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),**details)
    return control_json(path,previous,critical=False)


def check_control_settings(args):
    """Match actual new-D training, except the intended cosine weighting rule."""
    from new_d_control_utils import check_training_settings
    check_training_settings(args)
    source = read_json(args.matched_new_d_run/'model_manifest.json')
    state = read_json(args.matched_new_d_run/'run_status.json')
    if (source['model_spec']['matching'] != 'learned' or args.matching != 'cosine'
            or digest(state['selected_checkpoint']) != WEIGHT_SHA):
        raise ValueError('Control must match the actual learned new-D checkpoint used by H.')
    for key in ('hidden_dim','top_k','match_chunk'):
        if getattr(args,key) != source['train_options'][key]:
            raise ValueError(f'Matching architecture changed beyond weighting: {key}')
    if args.top_k != 8 or args.epochs != 15:
        raise ValueError('Experiment I uses the completed new-D Top8 and 15-epoch budget.')
    from rma_scratch_common import protocol
    current=vars(protocol(args))
    for key in ('features_list','image_size','n_ctx','vl_reduction','pq_mid_dim','pq_context',
                'visual_learner','textual_learner','pq_learner','fusion_type','sigma','batch_size'):
        if current[key] != source['base_protocol'][key]:
            raise ValueError(f'Training base protocol differs from new D: {key}')


def check_control_inputs(args, output):
    """Check split/reference pools now and every query/support episode in training."""
    source = args.matched_new_d_run
    for name in INPUT_FILES[:3]:
        if read_json(source/name) != read_json(output/name):
            raise ValueError(f'Control training inputs differ from new D: {name}')
    episodes = read_csv(source/'training_episodes.csv')
    manifest = read_json(source/'model_manifest.json')
    if not episodes or len(episodes) != args.epochs*manifest['train_query_count']:
        raise ValueError('Historical new-D training episode record is incomplete.')
    files = [source/name for name in (*INPUT_FILES,'model_manifest.json','run_status.json')]
    code_names=('rma_model.py','rma_scratch_common.py','rma_matching_control_utils.py','rma_joint_utils.py',
                'rma_utils.py','new_d_control_utils.py','train_rma_cosine.py','rma_control_io.py','rma_control_resume.py')
    files.extend(Path(__file__).parent/name for name in code_names)
    write_json(output/'matching_control_audit.json',dict(reference=str(source),
        reference_sha256={str(p):digest(p) for p in files},
        episode_policy='Each query/category/shot/ordered normal references must match new D; mismatch stops.',
        changed='Only Top8 candidate weighting: softmax(cosine + learned QK) -> softmax(cosine).',
        unchanged='Fresh four adapters, frozen CLIP, Q/V/relation heads, split, losses, optimizer, source selection.',
        key_projection='Retained in state layout but frozen and unused; Q remains trainable for relation features.',
        initialization='Original initialization order and RMA RNG restoration; no old adapter state loaded.',
        selected_epoch_policy='Independent source best among 15 epochs; never force epoch14.'))
    return episodes


def verify_control_inputs(output):
    for path,value in read_json(Path(output)/'matching_control_audit.json')['reference_sha256'].items():
        if digest(path) != value:
            raise ValueError(f'Historical training identity changed: {path}')
    resume=Path(output)/'resume_source.json'
    if resume.is_file():
        info=read_json(resume)
        for name,value in info['source_sha256'].items():
            if digest(Path(info['source'])/name)!=value:raise ValueError('Recovery source changed: '+name)


def verified_history(inputs):
    """Do not adopt modified metric tables as a fresh valid comparison reference."""
    folder=inputs['history'];status=read_json(folder/'run_status.json')
    expected=status.get('H_output_sha256',{}).get('metrics_combined.csv')
    if not expected or digest(folder/'metrics_combined.csv')!=expected:
        raise ValueError('Verified H/A metric table changed or lacks its original fingerprint.')
    return read_csv(folder/'metrics_combined.csv')


def control_checkpoint(checkpoint):
    """Require this control's completed, source-selected best checkpoint."""
    checkpoint=Path(checkpoint).resolve();folder=checkpoint.parent
    manifest,state=read_json(folder/'model_manifest.json'),read_json(folder/'run_status.json')
    if (state.get('complete') is not True or state.get('flow')!='new'
            or state.get('matched_cosine_weighting_control') is not True
            or state.get('four_adapter_gradient_checks_passed') is not True
            or manifest.get('matched_cosine_weighting_control') is not True
            or manifest.get('format')!='adaptclip-rma-scratch-v1'
            or manifest['model_spec']['matching']!='cosine'
            or manifest['model_spec']['top_k']!=8 or manifest['train_options']['epochs']!=15
            or checkpoint.name!='best.pth'
            or checkpoint != Path(state['selected_checkpoint']).resolve()):
        raise ValueError('Use completed experiment-I cosine-control training and its own best.pth.')
    history=read_csv(folder/'training_history.csv')
    if [int(r['epoch']) for r in history] != list(range(1,16)):
        raise ValueError('Complete 15-epoch source history required.')
    if any(not math.isfinite(float(r['validation_score'])) for r in history):
        raise ValueError('Nonfinite source validation score.')
    best=max(history,key=lambda r:float(r['validation_score']))
    if int(best['epoch'])!=state['best_epoch']:
        raise ValueError('Best checkpoint epoch does not match source selection.')
    verify_control_inputs(folder)
    return manifest,state,digest(checkpoint)


def output_table(candidate, history):
    """Compare one fixed candidate to both full historical H and original A."""
    previous={(r['Category'],r['Condition']):r for r in history}
    categories={r['Category'] for r in candidate}
    if categories!={c for c,k in previous if k=='H'} or categories!={c for c,k in previous if k=='baseline'}:
        raise ValueError('Candidate and H/A category coverage differ.')
    combined=list(candidate);deltas=[]
    for row in candidate:
        if row['Condition']!='I_cosine':raise ValueError('Wrong candidate condition.')
        for comparator in ('H','baseline'):
            old=previous[row['Category'],comparator]
            values={f:float(row[f])-float(old[f]) for f in FIELDS}
            if not all(math.isfinite(x) for x in values.values()):raise ValueError('Nonfinite six metrics.')
            deltas.append(dict(Category=row['Category'],Condition='I_cosine',Comparator=comparator,**values,
                ImageNet=sum(values[f] for f in FIELDS[:3]),PixelNet=sum(values[f] for f in FIELDS[3:]),Net=sum(values.values())))
    combined.extend(r for r in history if r['Condition'] in ('H','baseline'))
    return combined,deltas


def retention(rows, comparators=('H','baseline')):
    """Never pool shots or declare full retention from an incomplete campaign."""
    records=[]
    full={'Visa','BTAD','MPDD','SDD','DTD','part_glass'}
    for comparator in comparators:
        for shot in (1,2,4):
            part=[r for r in rows if r['Category']=='Mean' and r['Comparator']==comparator and int(r['Shot'])==shot]
            complete={r['Dataset'] for r in part}==full and len(part)==6
            positive=sum(float(r['Net'])>0 for r in part);net=sum(float(r['Net']) for r in part)
            records.append(dict(Comparator=comparator,Shot=shot,Complete=complete,Improved=positive,Net=net,
                                Pass=complete and positive>=3 and net>0))
    overall={c:dict(complete=all(r['Complete'] for r in records if r['Comparator']==c),
                   passed_shots=sum(r['Pass'] for r in records if r['Comparator']==c)) for c in comparators}
    for record in overall.values():record['retain']=record['complete'] and record['passed_shots']>=2
    return records,overall
