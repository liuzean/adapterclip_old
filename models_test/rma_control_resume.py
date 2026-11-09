"""Experiment-I epoch-boundary recovery; original failed run is never edited."""
import ast
import hashlib
import math
import os
import random
import shutil
from pathlib import Path

from fusion_ablation_utils import read_csv,read_json
from rma_control_io import write_json
from rma_utils import FIELDS,digest

ROOT=Path(__file__).resolve().parents[1]
# Captured from the actual pre-recovery implementation, whose file hash matches
# this interrupted run's original audit. These constrain legacy RNG replay.
LEGACY_TRAIN_SHA='bbf902e06d7cdacb2ae64d94b1f887e3b5ea082df94e776bb5842416be1dc900'
LEGACY_AST={
 'protocol':'6707a4027bf99b328ced28bd12c065fa10e73312e6a7e1fd4ab9b817b18bc14a',
 'initialize_fresh':'edebe4983f28d233cd0a6b0a64003a5884d4105146b8c63ea7ae340ddfbf998b',
 'training_losses':'74a9df1e74ed2271f1fcb653b16e75d3889844f52a0bad5180a80e9b1d110829',
 'checkpoint_selection':'0919ad9ee9a63ef8b13cd93306c5652b0d3e484f47e6a76950f99fad6b1e9c54',
 'source_validation':'90dc4be832e786836a1347e0f5acc5cd602d199231ff19bb0ecdd1cdc2450b0f',
 'train_batch_loop':'d91e00af9c76bd117197c45bca434cd32262d04dbf1b3e0576aac49b18ad478a'}


def inspect_source(source,args=None):
    """Read-only recovery preflight. Tensor checkpoint validation happens in child."""
    source=Path(source).resolve();source.relative_to((ROOT/'results/train_rma_cosine').resolve())
    manifest=read_json(source/'model_manifest.json');options=manifest['train_options']
    if (manifest.get('matched_cosine_weighting_control') is not True
            or manifest.get('flow')!='new' or manifest['model_spec']['matching']!='cosine'
            or options['epochs']!=15 or options['seed']!=10 or options['num_workers']!=0):
        raise ValueError('Recovery requires the matched experiment-I cosine training, 15 epochs and workers0.')
    history=read_csv(source/'training_history.csv');epochs=[int(x['epoch']) for x in history]
    if not epochs or epochs!=list(range(1,len(epochs)+1)) or not 1<=len(epochs)<15:
        raise ValueError('Recovery needs consecutive completed epochs before epoch15.')
    state=read_json(source/'run_status.json')
    if state.get('complete') or state.get('epoch')!=len(history):
        raise ValueError('Training status and completed epoch history disagree; do not guess recovery epoch.')
    if any(not math.isfinite(float(r['validation_score'])) for r in history):raise ValueError('Invalid source history.')
    if args is not None:
        for key in ('epochs','seed','batch_size','accumulation','learning_rate','weight_decay','num_workers',
                    'validation_fraction','hidden_dim','top_k','match_chunk','matching','betas','optimizer',
                    'train_reference_shots','fusion_weight','gradient_clip'):
            if getattr(args,key)!=options[key]:raise ValueError(f'Recovery changes training setting: {key}')
        for key in ('train_data_path','pretrained_model','baseline_checkpoint','matched_new_d_run'):
            if Path(getattr(args,key)).resolve()!=Path(options[key]).resolve():raise ValueError(f'Recovery path differs: {key}')
    audit=read_json(source/'matching_control_audit.json');old_hashes=audit['reference_sha256']
    for path,value in old_hashes.items():
        if Path(path).is_relative_to(ROOT/'results') and digest(path)!=value:
            raise ValueError(f'Historical matching input changed: {path}')
    for name in ('rma_model.py','rma_joint_utils.py','rma_utils.py','new_d_control_utils.py','train_rma_cosine.py'):
        path=ROOT/'models_test'/name
        if old_hashes.get(str(path))!=digest(path):raise ValueError(f'Numerical dependency changed: {name}')
    files=['last.pth','best.pth','training_history.csv','training_steps.csv','training_episodes.csv',
           'validation_metrics.csv','source_split.json','training_queries.json','training_reference_pool.json',
           'model_manifest.json','experiment_config.json','matching_control_audit.json']
    files += [f'epoch_{i}.pth' for i in epochs]
    if any(not (source/name).is_file() for name in files):raise FileNotFoundError('Incomplete recovery checkpoint/history files.')
    count=manifest['train_query_count'];episodes=read_csv(source/'training_episodes.csv')
    if len(episodes)!=len(history)*count:raise ValueError('Incomplete saved training episode prefix.')
    expected=read_csv(Path(options['matched_new_d_run'])/'training_episodes.csv')
    if episodes!=expected[:len(episodes)]:raise ValueError('Saved control episodes differ from learned new D.')
    steps=read_csv(source/'training_steps.csv')
    if sum(int(r['batch_size']) for r in steps)!=len(episodes):raise ValueError('Training step/query coverage mismatch.')
    best=max(history,key=lambda r:float(r['validation_score']))
    return dict(source=str(source),completed_epoch=len(history),next_epoch=len(history)+1,
        best_epoch=int(best['epoch']),best_score=float(best['validation_score']),source_train_sha=old_hashes.get(str(ROOT/'models_test/rma_scratch_common.py')),
        source_sha256={name:digest(source/name) for name in files})


def fork_source(source,output,args):
    info=inspect_source(source,args);source=Path(info['source']);output=Path(output)
    # Preserve weights/history under a fresh training directory. Do not copy the
    # failed run_status or fixed-name .tmp left by the old logging failure.
    excluded={'model_manifest.json','experiment_config.json','matching_control_audit.json'}
    for name,expected in info['source_sha256'].items():
        if name not in excluded:
            shutil.copy2(source/name,output/name)
            if digest(output/name)!=expected:raise ValueError('Recovery copy verification failed: '+name)
    write_json(output/'resume_source.json',info)
    return info


def capture_rng(support_rng):
    """Save weights_only-safe Python/NumPy/CPU/CUDA/support RNG state."""
    import numpy as np
    import torch
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=n[0],keys=n[1].tolist(),position=int(n[2]),
                has_gauss=int(n[3]),cached_gaussian=float(n[4])),torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all(),support=support_rng.getstate())


def restore_rng(state,support_rng):
    import numpy as np
    import torch
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda'])
    support_rng.setstate(state['support'])


def check_legacy_code(info):
    if info['source_train_sha']!=LEGACY_TRAIN_SHA:
        raise ValueError('Unknown legacy implementation without RNG snapshots; cannot safely reconstruct it.')
    tree=ast.parse((ROOT/'models_test/rma_scratch_common.py').read_text(encoding='utf-8-sig'))
    for name,expected in LEGACY_AST.items():
        if name=='train_batch_loop':
            node=next(x for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Tuple)
                      and [ast.unparse(y) for y in x.target.elts]==['step','items'])
        else:node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)
        actual=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        if actual!=expected:raise ValueError('Legacy numerical/replay contract changed: '+name)


def replay_sampling(loader,val_loader,pool,saved,completed_epoch,shot_cycle,rng,progress=None):
    """Replay only CPU metadata and RNG draws, never model forward/backward.

    DataLoader creation/iteration consumes base seeds; use the same public
    batch sampler/generator/workers0 semantics for training and validation.
    Every old query/reference is checked before proceeding to future epochs.
    """
    import torch
    class MetadataOnly(torch.utils.data.Dataset):
        def __init__(self,data):self.rows,self.root=data.data_all,data.root
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            r=self.rows[index];return dict(img_path=os.path.join(self.root,r['img_path']),cls_name=r['cls_name'])
    train=torch.utils.data.DataLoader(MetadataOnly(loader.dataset),batch_sampler=loader.batch_sampler,
                                    num_workers=0,generator=loader.generator)
    val=torch.utils.data.DataLoader(MetadataOnly(val_loader.dataset),batch_sampler=val_loader.batch_sampler,
                                  num_workers=0,generator=val_loader.generator)
    count=steps=0
    for epoch in range(1,completed_epoch+1):
        if progress:progress('reconstructing_sampling',epoch=epoch,batch=0)
        for step,items in enumerate(train):
            shot=shot_cycle[steps%len(shot_cycle)]
            for path,category in zip(items['img_path'],items['cls_name']):
                chosen=rng.sample(pool[category],shot)
                row=dict(epoch=str(epoch),step=str(step+1),query=path,category=category,shot=str(shot),
                         references=' | '.join(r['img_path'] for r in chosen))
                if count>=len(saved) or row!={k:str(v) for k,v in saved[count].items()}:
                    raise ValueError('Legacy query/reference RNG replay mismatch; no training continued.')
                count+=1
            steps+=1
        for _ in (1,2,4):
            for _items in val:pass
    if count!=len(saved):raise ValueError('Legacy replay did not cover every saved episode.')
    return dict(episodes=count,batches=steps,model_forward_backward_replayed=False)


def restore_training(args,output,adapters,rma,optimizer,model,loader,val_loader,split,support_rng,progress):
    """Restore complete epoch state and retain best selection across all 15 epochs."""
    import torch
    info=read_json(output/'resume_source.json')
    for name,value in info['source_sha256'].items():
        if digest(Path(info['source'])/name)!=value:raise ValueError('Resume source changed: '+name)
    payload=torch.load(output/'last.pth',map_location='cpu',weights_only=True)
    if (payload.get('epoch')!=info['completed_epoch'] or 'optimizer' not in payload
            or payload.get('matched_cosine_weighting_control') is not True or payload['model_spec']!=rma.spec
            or payload['best_epoch']!=info['best_epoch'] or abs(payload['best_score']-info['best_score'])>1e-9):
        raise ValueError('last.pth disagrees with the complete source epoch/history.')
    for key,module in zip(('textual_learner','visual_learner','pq_learner','rma'),(*adapters,rma)):
        module.load_state_dict(payload[key],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    best_payload=torch.load(output/'best.pth',map_location='cpu',weights_only=True)
    if best_payload['epoch']!=info['best_epoch'] or abs(best_payload['validation_score']-info['best_score'])>1e-9:
        raise ValueError('Copied best.pth does not match the historical source-best epoch.')
    del best_payload
    history=read_csv(output/'training_history.csv');steps=read_csv(output/'training_steps.csv')
    episodes=read_csv(output/'training_episodes.csv');validation=read_csv(output/'validation_metrics.csv')
    for rows in (history,steps):
        for row in rows:
            for key,value in list(row.items()):
                row[key]=int(value) if key in ('epoch','step','shot','batch_size') else float(value)
    for row in episodes:
        for key in ('epoch','step','shot'):row[key]=int(row[key])
    for row in validation:
        row['epoch']=int(row['epoch']);row['shot']=int(row['shot'])
        for key in FIELDS:row[key]=float(row[key])
    if payload.get('resume_rng'):
        restore_rng(payload['resume_rng'],support_rng);replay=dict(mode='saved_rng_snapshot')
    else:
        check_legacy_code(info)
        print(f"回放前{info['completed_epoch']}轮采样编号（仅CPU元数据，不重复模型训练）；逐条核对查询和参考。",flush=True)
        if any(m.training for m in model.modules()):raise ValueError('Legacy CLIP must remain eval.')
        for module in (*adapters,rma):
            for child in module.modules():
                if isinstance(child,torch.nn.modules.dropout._DropoutNd) and child.p>0:
                    raise ValueError('Legacy replay cannot reconstruct active dropout RNG.')
                if isinstance(child,(torch.nn.MultiheadAttention,torch.nn.RNNBase)) and child.dropout>0:
                    raise ValueError('Legacy replay cannot reconstruct attention/RNN dropout RNG.')
        if args.num_workers!=0:raise ValueError('Legacy reconstruction only verified for workers0.')
        for data in (loader.dataset,val_loader.dataset):
            if getattr(data,'mode',None)!='test':raise ValueError('Legacy replay requires deterministic source-query mode.')
            transforms=[getattr(data,'transform',None),getattr(data,'target_transform',None)]
            while transforms:
                transform=transforms.pop()
                if transform is None:continue
                name=type(transform).__name__
                if name.startswith('Random') or name in ('ColorJitter','AutoAugment','RandAugment','AugMix'):
                    raise ValueError('Legacy replay cannot reconstruct stochastic image transforms.')
                transforms.extend(getattr(transform,'transforms',()))
        replay=replay_sampling(loader,val_loader,split['training_support_pool'],episodes,
                               info['completed_epoch'],args.train_reference_shots,support_rng,progress)
        replay['mode']='legacy_metadata_sampler_reconstruction'
    # A further failure during epoch13 can now resume from this fork's epoch12
    # boundary with a real RNG snapshot, instead of reconstructing legacy again.
    from rma_utils import atomic_checkpoint
    payload['resume_rng']=capture_rng(support_rng)
    atomic_checkpoint(output/'last.pth',payload)
    del payload
    write_json(output/'resume_audit.json',dict(**info,recovery=replay,
        original_source_untouched=True,history_included_in_best_selection=True,
        bitwise_uninterrupted_training_equivalence_verified=False))
    return dict(history=history,steps=steps,episodes=episodes,validation_rows=validation,
                best_value=info['best_score'],best_epoch=info['best_epoch'],first_epoch=info['next_epoch'])
