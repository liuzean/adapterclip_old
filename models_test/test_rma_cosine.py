r"""实验I测试：新训练cosine对照，固定H公式，历史H为直接对照、原A为共同基线。

默认六集依次各1/2/4-shot、seed10，每条件独立进程。历史只作身份核对/评价。
示例 python .\models_test\test_rma_cosine.py --checkpoint <本次训练best.pth>
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import new_directory,path_key,project_path,read_csv,read_json,write_csv
from h_inference import WEIGHT_SHA,recorded_inputs
from rma_matching_control_utils import control_checkpoint,output_table,retention,verified_history
from rma_control_io import write_json,write_progress
from rma_utils import FIELDS,cleanup,digest,record_pipeline_progress,run_child


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--checkpoint',default='results/train_rma_cosine/mvtec_seed10/best.pth')
    p.add_argument('--reference_run',default='results/rma_score_coordination_inference/experiment_H_inference_seed10')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--shots',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--status');p.add_argument('--resume_from')
    for key in ('_progress_file','_session','_output','_dataset','_shot'):p.add_argument('--'+key,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.datasets!=sorted(set(a.datasets),key=DATASETS.index) or a.shots!=sorted(set(a.shots)) or a.num_workers<0:
        p.error('Ordered unique datasets/shots and nonnegative workers required.')
    hidden=[getattr(a,k) is not None for k in ('_session','_output','_dataset','_shot')]
    if any(hidden) and not all(hidden):p.error('Incomplete worker arguments.')
    if a.status and (a.resume_from or any(hidden)):p.error('Status is read-only and separate.')
    tokens=sys.argv[1:] if argv is None else argv
    if a.resume_from and any(t.startswith('--') and t.split('=',1)[0]!='--resume_from' for t in tokens):
        p.error('Resume restores all saved settings; no overrides.')
    for name in ('checkpoint','reference_run'):setattr(a,name,project_path(getattr(a,name),ROOT))
    return a


def code_hashes():
    names=('test_rma_cosine.py','rma_matching_control_utils.py','rma_control_io.py','rma_control_resume.py','h_inference.py','rma_model.py',
      'rma_scratch_common.py','rma_joint_utils.py','rma_utils.py','fusion_ablation.py','fusion_ablation_utils.py',
      'branch_diagnostics_utils.py','rma_fixed_fusion_utils.py','rma_output_ablation_utils.py',
      'rma_score_coordination.py','rma_score_coordination_utils.py')
    return {str(ROOT/'models_test'/n):digest(ROOT/'models_test'/n) for n in names}


def verify_files(config):
    if code_hashes()!=config['code_sha256']:raise ValueError('Control code changed during campaign.')
    for path,value in config['input_sha256'].items():
        if digest(path)!=value:raise ValueError(f'Control input changed: {path}')


def worker(args):
    import torch
    from h_inference import load_model,predict
    from rma_utils import MetricsBuffer,check_frozen,save_tables
    from tqdm import tqdm
    session=project_path(args._session,ROOT);session.relative_to((ROOT/'results/test_rma_cosine').resolve())
    output=project_path(args._output,ROOT);output.relative_to(session)
    output.mkdir(parents=True,exist_ok=False)
    config=read_json(session/'campaign_config.json');shot=int(args._shot)
    write_progress(output/'run_status.json',dict(complete=False,phase='initializing',processed_queries=0))
    try:
        verify_files(config);inputs=recorded_inputs(args.reference_run,args._dataset,shot)
        write_progress(output/'run_status.json',dict(complete=False,phase='loading_weights',processed_queries=0,
                   total_queries=len(inputs['manifest']),dataset=args._dataset,shot=shot))
        write_json(output/'experiment_config.json',dict(inputs['config'],result_dir=str(output),
            experiment='I_cosine',candidate_checkpoint=str(args.checkpoint),candidate_epoch=config['selected_epoch'],
            candidate_checkpoint_sha256=config['checkpoint_sha256'],matching='cosine',coefficient=.25,local_fraction=.125,
            original_protocol_checkpoint_role='historical A/H identity only; candidate loaded separately'))
        write_csv(output/'references.csv',list(inputs['reference_paths'][0]),inputs['reference_paths'])
        write_csv(output/'sample_manifest.csv',list(inputs['manifest'][0]),inputs['manifest'])
        state=load_model(args.checkpoint,inputs,output,args.num_workers,expected_sha=config['checkpoint_sha256'],
                         expected_epoch=config['selected_epoch'],expected_matching='cosine')
        p_rows,i_rows,scores=[],[],[];buffer=None;closed=set();count=0
        def flush():
            p,i=buffer.finish(state.device);p_rows.extend(p);i_rows.extend(i);cleanup()
        with torch.no_grad():
            for items in tqdm(state.loader,desc=f'I cosine {args._dataset} {shot}-shot'):
                write_progress(output/'run_status.json',dict(complete=False,phase='evaluating',processed_queries=count,
                    total_queries=len(inputs['manifest']),dataset=args._dataset,shot=shot,
                    current_categories=list(dict.fromkeys(items['cls_name']))))
                maps,chosen,parts=predict(state,items['img'],items['cls_name'],items['img_path'],'H')
                if maps.shape!=items['img_mask'][:,0].shape or chosen.shape!=(len(items['cls_name']),):raise ValueError('Wrong output shape.')
                if not torch.isfinite(maps).all() or not torch.isfinite(chosen).all():raise ValueError('Nonfinite prediction.')
                for i,category in enumerate(items['cls_name']):
                    expected=inputs['manifest'][count];path=items['img_path'][i];label=int(items['anomaly'][i])
                    if category!=expected['Category'] or path_key(path)!=path_key(expected['Image Path']) or label!=int(expected['Sample Type']=='Anomaly'):
                        raise ValueError('Control query identity/order changed.')
                    if buffer is None or buffer.category!=category:
                        if buffer is not None:flush();closed.add(buffer.category)
                        if category in closed:raise ValueError('Noncontiguous category order.')
                        buffer=MetricsBuffer(category,conditions=('I_cosine',))
                    buffer.append(items['img_mask'][i,0]>.5,label,{'I_cosine':maps},{'I_cosine':chosen},i)
                    scores.append(dict(Category=category,**{'Image Path':path},Label=label,Score=float(chosen[i]),
                                       SE_candidate=float(parts['E'][i]),G0_candidate=float(parts['G0'][i])))
                    count+=1
                write_progress(output/'run_status.json',dict(complete=False,phase='evaluating',processed_queries=count,
                    total_queries=len(inputs['manifest']),dataset=args._dataset,shot=shot,
                    current_categories=list(dict.fromkeys(items['cls_name']))))
                del maps,chosen,parts
        if buffer is None or count!=len(inputs['manifest']):raise ValueError('Incomplete control coverage.')
        flush();candidate=save_tables(output,p_rows,i_rows)
        combined,deltas=output_table(candidate,verified_history(inputs))
        write_csv(output/'metrics_combined.csv',['Category','Condition',*FIELDS],combined)
        write_csv(output/'dataset_deltas.csv',list(deltas[0]),deltas)
        write_csv(output/'image_scores.csv',list(scores[0]),scores)
        for snapshot in state.snapshots:check_frozen(snapshot)
        verify_files(config)
        files=('metrics_combined.csv','dataset_deltas.csv','image_scores.csv','references.csv','sample_manifest.csv','experiment_config.json')
        write_json(output/'run_status.json',dict(complete=True,phase='completed',processed_queries=count,
          total_queries=count,dataset=args._dataset,shot=shot,frozen_integrity_passed=True,
          matching='cosine',reference_identity_passed=True,formula='fixed H; own candidate scores and maps',
          output_sha256={n:digest(output/n) for n in files}))
    except BaseException as error:
        try:prior=read_json(output/'run_status.json')
        except (OSError,ValueError):prior={}
        write_progress(output/'run_status.json',dict(prior,complete=False,error=str(error) or 'Ctrl+C'));raise
    finally:cleanup()


def summarize(session,state,args):
    combined=[];deltas=[]
    for key,folder in state['job_results'].items():
        d,s=key.split('/');folder=Path(folder)
        combined.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(folder/'metrics_combined.csv'))
        deltas.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(folder/'dataset_deltas.csv'))
    write_csv(session/'metrics_summary.csv',list(combined[0]),combined)
    write_csv(session/'dataset_deltas.csv',list(deltas[0]),deltas)
    records,overall=retention(deltas)
    write_csv(session/'retention_by_shot.csv',list(records[0]),records)
    learned=[]
    for row in deltas:
        if row['Comparator']=='H':
            learned.append(dict(row,Condition='H',Comparator='I_cosine',
                **{f:-float(row[f]) for f in (*FIELDS,'ImageNet','PixelNet','Net')}))
    write_csv(session/'learned_H_minus_control_deltas.csv',list(learned[0]),learned)
    learned_records,learned_overall=retention(learned,comparators=('I_cosine',))
    write_csv(session/'learned_H_retention_by_shot.csv',list(learned_records[0]),learned_records)
    write_json(session/'retention_overall.json',dict(primary_comparator='H',secondary_comparator='original A',
        direction='I cosine minus comparator; negative favors learned H',comparisons=overall,
        learned_H_vs_control=learned_overall,
        note='Ablation evidence does not prove correspondence correctness or universal optimality.'))


def main(argv=None):
    a=parse_args(argv)
    if a.status:
        folder=project_path(a.status,ROOT);s=read_json(folder/'run_status.json');print(folder,s)
        if s.get('current_output') and (Path(s['current_output'])/'run_status.json').is_file():print(read_json(Path(s['current_output'])/'run_status.json'))
        return
    if a._session:worker(a);return
    if a.resume_from:
        session=project_path(a.resume_from,ROOT);session.relative_to((ROOT/'results/test_rma_cosine').resolve())
        config=read_json(session/'campaign_config.json')
        for k,v in config['options'].items():setattr(a,k,project_path(v,ROOT) if k in ('checkpoint','reference_run') else v)
        a._progress_file=config.get('progress_file')
        verify_files(config);state=read_json(session/'run_status.json')
    else:
        manifest,train,weight_sha=control_checkpoint(a.checkpoint)
        files={str(a.checkpoint):weight_sha}
        for name in ('model_manifest.json','run_status.json','training_history.csv','matching_control_audit.json'):
            p=a.checkpoint.parent/name;files[str(p)]=digest(p)
        for d in a.datasets:
            for s in a.shots:
                inputs=recorded_inputs(a.reference_run,d,s)
                verified_history(inputs)
                if digest(inputs['config']['pretrained_model'])!=manifest['pretrained_sha256'] or inputs['config']['checkpoint_sha256']!=WEIGHT_SHA:
                    raise ValueError('Candidate CLIP or historical H identity differs.')
                for p in [inputs['meta'],Path(inputs['config']['pretrained_model']),*[inputs['history']/n for n in
                          ('experiment_config.json','input_audit.json','references.csv','sample_manifest.csv','run_status.json','metrics_combined.csv')]]:
                    files[str(p)]=digest(p)
        opts={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if not k.startswith('_') and k not in ('status','resume_from')}
        config=dict(options=opts,checkpoint_sha256=weight_sha,selected_epoch=train['best_epoch'],matching='cosine',
                    code_sha256=code_hashes(),input_sha256=files,coefficient=.25,local_fraction=.125,seed=10,
                    progress_file=str(project_path(a._progress_file,ROOT)) if a._progress_file else None)
        session=new_directory(ROOT/'results/test_rma_cosine','I_cosine_seed10');write_json(session/'campaign_config.json',config)
        state=dict(complete=False,current=None,current_output=None,completed_jobs=[],job_results={});write_json(session/'run_status.json',state)
    receipt=project_path(a._progress_file,ROOT) if a._progress_file else None
    record_pipeline_progress(receipt,'evaluating',session=str(session),evaluation_jobs=[(d,s) for d in a.datasets for s in a.shots])
    try:
        for d in a.datasets:
            for s in a.shots:
                key=f'{d}/{s}';old=state['job_results'].get(key)
                if old and (Path(old)/'run_status.json').is_file():
                    saved=read_json(Path(old)/'run_status.json')
                    if saved.get('complete'):
                        for name,value in saved['output_sha256'].items():
                            if digest(Path(old)/name)!=value:raise ValueError('Completed output changed.')
                        if key not in state['completed_jobs']:state['completed_jobs'].append(key)
                        continue
                parent=session/d;parent.mkdir(exist_ok=True)
                output=parent/f'{s}shot_seed10';index=2
                while output.exists():output=parent/f'{s}shot_seed10_run{index:02d}';index+=1
                state.update(complete=False,current=key,current_output=str(output));state['job_results'][key]=str(output);write_json(session/'run_status.json',state)
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),'--_output',str(output),'--_dataset',d,'--_shot',str(s)]
                for k,v in config['options'].items():cmd.extend(['--'+k,*map(str,v)] if isinstance(v,list) else ['--'+k,str(v)])
                print(f'I cosine {key}; independent process; output={output}',flush=True)
                try:
                    run_child(cmd)
                    if read_json(output/'run_status.json').get('complete') is not True:raise ValueError('Control worker incomplete.')
                finally:cleanup()
                state['completed_jobs'].append(key);write_json(session/'run_status.json',state)
        summarize(session,state,a);verify_files(config)
        state.update(complete=True,current=None,current_output=None);state.pop('error',None);write_json(session/'run_status.json',state)
        record_pipeline_progress(receipt,'completed',session=str(session));print('Control evaluation completed:',session)
    except BaseException as error:
        state.update(complete=False,error=str(error) or 'Ctrl+C');write_json(session/'run_status.json',state)
        print(f"Stopped at {state['current']}; completed={state['completed_jobs']}\nSession={session}\nWorker={state['current_output']}",flush=True);raise


if __name__=='__main__':main()
