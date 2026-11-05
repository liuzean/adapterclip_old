"""Metadata and saved-table checks for the matched new-D three-adapter control.

No torch imports: usable by parent schedulers without allocating GPU memory.
"""
from pathlib import Path

from fusion_ablation_utils import read_csv, read_json, write_csv, write_json
from rma_utils import FIELDS, digest


def check_training_settings(args):
    """Match the actual completed new-D run, rather than only assumed defaults."""
    source = args.matched_new_d_run
    manifest, state = read_json(source/'model_manifest.json'), read_json(source/'run_status.json')
    if (manifest.get('format') != 'adaptclip-rma-scratch-v1' or manifest.get('flow') != 'new'
            or state.get('complete') is not True):
        raise ValueError('matched_new_d_run must be a completed four-adapter new-D training.')
    for name in ('epochs','batch_size','accumulation','learning_rate','weight_decay','num_workers',
                 'validation_fraction','betas','optimizer','train_reference_shots','fusion_weight','gradient_clip','seed'):
        if getattr(args,name) != manifest['train_options'][name]:
            raise ValueError(f'Matched new-D training setting differs: {name}')
    if (digest(args.pretrained_model) != manifest['pretrained_sha256']
            or digest(args.baseline_checkpoint) != manifest['base_sha256']
            or digest(args.train_data_path/'meta.json') != manifest['source_meta_sha256']):
        raise ValueError('CLIP, original A identity or source metadata differ from new D.')
    if args.train_data_path.resolve() != Path(manifest['train_options']['train_data_path']).resolve():
        raise ValueError('Use the identical new-D source dataset directory.')


def check_training_inputs(args, output):
    """Require identical split, queries, reference pool and all planned episodes."""
    evidence = {}
    for filename in ('source_split.json','training_queries.json','training_reference_pool.json'):
        if read_json(output/filename) != read_json(args.matched_new_d_run/filename):
            raise ValueError(f'Matched new-D input differs: {filename}')
        evidence[filename] = digest(args.matched_new_d_run/filename)
    episodes = read_csv(args.matched_new_d_run/'training_episodes.csv')
    if not episodes:
        raise ValueError('New-D episode identities are required for the matched control.')
    manifest = read_json(args.matched_new_d_run/'model_manifest.json')
    if len(episodes) != args.epochs * manifest['train_query_count']:
        raise ValueError('Incomplete new-D training episode record.')
    write_json(output/'matched_new_D_audit.json', dict(reference=str(args.matched_new_d_run),
        source_files=evidence, reference_episodes_sha256=digest(args.matched_new_d_run/'training_episodes.csv'),
        episode_policy='Every sampled query/reference episode must match historical new D; mismatch stops training.',
        initialization='Same CLIP and three fresh adapters; RMA is never instantiated.',
        removed='RMA parameters, RMA branch losses; fusion changes from four branches to three.'))
    return episodes


def check_evaluation_reference(args, jobs, audit):
    source = args.new_d_run
    state, campaign = read_json(source/'run_status.json'), read_json(source/'campaign_config.json')
    if state.get('complete') is not True or campaign.get('flow') != 'new' or campaign.get('candidate') != 'S_new':
        raise ValueError('new_d_run must be a completed full new-D evaluation.')
    manifest = read_json(args.checkpoint.parent/'model_manifest.json')
    matched = Path(manifest['train_options']['matched_new_d_run']).resolve()
    reference_checkpoint = Path(campaign['checkpoint']).resolve()
    training_state = read_json(matched/'run_status.json')
    if (reference_checkpoint.parent != matched or reference_checkpoint != Path(training_state['selected_checkpoint']).resolve()
            or digest(reference_checkpoint) != campaign['checkpoint_sha256']):
        raise ValueError('Evaluation new D is not the matched training best checkpoint.')
    for dataset,shot in jobs:
        directory = source/dataset/f'{shot}shot_seed10'
        s = read_json(directory/'run_status.json')
        if not all(s.get(k) is True for k in ('complete','baseline_parity_passed','frozen_integrity_passed')):
            raise ValueError(f'Incomplete or unverified new-D round: {dataset}/{shot}')
        if read_json(directory/'input_audit.json') != audit[dataset][str(shot)]:
            raise ValueError('Target queries/references/protocol differ from new D.')


def compare_to_new_d(args, session, jobs, candidate):
    """Save control-minus-new-D deltas without hiding the separate original-A comparison."""
    records=[]
    for dataset,shot in jobs:
        path = Path(dataset)/f'{shot}shot_seed10'/'metrics_combined.csv'
        current = read_csv(session/path)
        previous = read_csv(args.new_d_run/path)
        a = next(r for r in current if r['Category']=='Mean' and r['Condition']==candidate)
        b = next(r for r in previous if r['Category']=='Mean' and r['Condition']=='S_new')
        ca = next(r for r in current if r['Category']=='Mean' and r['Condition']=='baseline')
        cb = next(r for r in previous if r['Category']=='Mean' and r['Condition']=='baseline')
        if any(abs(float(ca[f])-float(cb[f]))>1e-8 for f in FIELDS):
            raise ValueError('Original-A baselines differ between three-control and new D.')
        delta={f:float(a[f])-float(b[f]) for f in FIELDS}
        records.append(dict(Dataset=dataset,Shot=shot,Condition=candidate,Comparator='S_new',**delta,Net=sum(delta.values())))
    write_csv(session/'deltas_vs_new_D.csv',list(records[0]),records)
    write_json(session/'comparison_reference.json',dict(primary='original A',secondary='new D full model',
        new_d_run=str(args.new_d_run),direction='three-adapter control minus new D; negative favors new D',
        explanation='Retrained without RMA from initialization; not inference-only removal of RMA.'))
