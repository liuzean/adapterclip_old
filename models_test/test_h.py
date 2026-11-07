r"""精简H/E独立测试入口；原test.py不变。固定epoch14、lambda=.25、局部12.5%。

Root: python .\models_test\test_h.py ; models_test: python .\test_h.py
Default six datasets, each1/2/4-shot, seed10; process exits before next round.
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import new_directory,path_key,project_path,read_csv,read_json,write_csv
from h_inference import WEIGHT_SHA,recorded_inputs
from rma_score_coordination import write_json
from rma_utils import FIELDS,cleanup,digest,record_pipeline_progress,run_child


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('H','E'),default='H')
    p.add_argument('--checkpoint',default='results/train_rma_scratch_new/mvtec_seed10/best.pth')
    p.add_argument('--reference_run',default='results/rma_score_coordination_inference/experiment_H_inference_seed10')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--shots',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--verify_history',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--status');p.add_argument('--resume_from')
    p.add_argument('--_progress_file',help=argparse.SUPPRESS)
    p.add_argument('--_session',help=argparse.SUPPRESS);p.add_argument('--_output',help=argparse.SUPPRESS)
    p.add_argument('--_dataset',choices=DATASETS,help=argparse.SUPPRESS);p.add_argument('--_shot',type=int,choices=(1,2,4),help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.datasets!=sorted(set(args.datasets),key=DATASETS.index) or args.shots!=sorted(set(args.shots)) or args.num_workers<0:
        p.error('Ordered unique datasets/shots and nonnegative workers required.')
    hidden=[getattr(args,k) is not None for k in ('_session','_output','_dataset','_shot')]
    if any(hidden) and not all(hidden):p.error('Incomplete worker fields.')
    tokens=sys.argv[1:] if argv is None else argv
    if args.status and args.resume_from:p.error('Use status or resume separately.')
    if args.resume_from and any(t.startswith('--') and t.split('=',1)[0]!='--resume_from' for t in tokens):p.error('Resume restores saved settings.')
    for name in ('checkpoint','reference_run'):setattr(args,name,project_path(getattr(args,name),ROOT))
    return args


def code_hashes():
    names=('h_inference.py','fusion_ablation.py','fusion_ablation_utils.py','branch_diagnostics_utils.py',
           'rma_fixed_fusion_utils.py','rma_output_ablation_utils.py','rma_model.py','rma_score_coordination_utils.py',
           'rma_score_coordination.py','rma_utils.py','rma_scratch_common.py','rma_joint_utils.py')
    entry=Path(__file__).resolve()
    return {str(entry):digest(entry),**{str(ROOT/'models_test'/n):digest(ROOT/'models_test'/n) for n in names}}


def verify_files(config):
    if config['code_sha256']!=code_hashes():raise ValueError('Code changed during this run.')
    for path,value in config['input_sha256'].items():
        if digest(path)!=value:raise ValueError(f'Input/weight changed: {path}')


def worker(args):
    import math
    import torch
    from h_inference import load_model,predict
    from rma_utils import MetricsBuffer,check_frozen,save_tables
    from tqdm import tqdm
    session=project_path(args._session,ROOT);session.relative_to((ROOT/'results'/'test_h').resolve())
    output=project_path(args._output,ROOT);output.relative_to(session)
    output.mkdir(parents=True,exist_ok=False)
    config=read_json(session/'campaign_config.json')
    write_json(output/'run_status.json',dict(complete=False,phase='initializing',processed_queries=0))
    try:
        verify_files(config);inputs=recorded_inputs(args.reference_run,args._dataset,args._shot)
        write_json(output/'experiment_config.json',dict(inputs['config'],mode=args.mode,checkpoint=str(args.checkpoint),
            epoch=14,coefficient=.25,rma_local_fraction=.125,result_dir=str(output),history_verification=args.verify_history))
        write_csv(output/'references.csv',list(inputs['reference_paths'][0]),inputs['reference_paths'])
        write_csv(output/'sample_manifest.csv',list(inputs['manifest'][0]),inputs['manifest'])
        state=load_model(args.checkpoint,inputs,output,args.num_workers)
        history=read_csv(inputs['history']/'image_scores.csv') if args.verify_history else []
        pixel_rows,image_rows,scores,checks=[],[],[],[];buffer=None;closed=set();count=0
        def flush():
            p,i=buffer.finish(state.device);pixel_rows.extend(p);image_rows.extend(i)
            save_tables(output,pixel_rows,image_rows);cleanup()
        with torch.no_grad():
            for items in tqdm(state.loader,desc=f'{args.mode} {args._dataset} {args._shot}-shot'):
                maps,chosen,parts=predict(state,items['img'],items['cls_name'],items['img_path'],args.mode)
                if not torch.isfinite(maps).all() or not torch.isfinite(chosen).all():raise ValueError('Nonfinite prediction.')
                if maps.shape!=items['img_mask'][:,0].shape or chosen.shape!=(len(items['cls_name']),):raise ValueError('Output shape mismatch.')
                for i,category in enumerate(items['cls_name']):
                    path=items['img_path'][i];label=int(items['anomaly'][i]);expected=inputs['manifest'][count]
                    if category!=expected['Category'] or path_key(path)!=path_key(expected['Image Path']) or label!=int(expected['Sample Type']=='Anomaly'):raise ValueError('Query order/identity changed.')
                    if buffer is None or category!=buffer.category:
                        if buffer is not None:flush();closed.add(buffer.category)
                        if category in closed:raise ValueError('Noncontiguous categories.')
                        buffer=MetricsBuffer(category,conditions=(args.mode,))
                    buffer.append(items['img_mask'][i,0]>.5,label,{args.mode:maps},{args.mode:chosen},i)
                    scores.append(dict(Category=category,**{'Image Path':path},Label=label,Score=float(chosen[i]),E=float(parts['E'][i]),G0=float(parts['G0'][i])))
                    if args.verify_history:
                        old=history[count]
                        if old['Category']!=category or path_key(old['Image Path'])!=path_key(path) or int(old['Label'])!=label:raise ValueError('History query identity mismatch.')
                        delta=float(chosen[i])-float(old[f'{args.mode} Image Score'])
                        checks.append(dict(Image=path,Delta=delta,Passed=math.isfinite(delta) and abs(delta)<=1e-6))
                        if not checks[-1]['Passed']:raise ValueError('Compact entry historical score mismatch.')
                    count+=1
                write_json(output/'run_status.json',dict(complete=False,phase='evaluating',processed_queries=count))
                del maps,chosen,parts
        if buffer is None or count!=len(inputs['manifest']):raise ValueError('Incomplete coverage.')
        flush();table=save_tables(output,pixel_rows,image_rows)
        write_csv(output/'image_scores.csv',list(scores[0]),scores)
        metric_checks=[]
        if args.verify_history:
            old={(r['Category'],r['Condition']):r for r in read_csv(inputs['history']/'metrics_combined.csv')}
            for row in table:
                for field in FIELDS:
                    delta=float(row[field])-float(old[row['Category'],args.mode][field])
                    metric_checks.append(dict(Category=row['Category'],Metric=field,DeltaPP=delta,Passed=math.isfinite(delta) and abs(delta)<=.001))
            write_csv(output/'score_checks.csv',list(checks[0]),checks)
            write_csv(output/'metric_checks.csv',list(metric_checks[0]),metric_checks)
            if not all(r['Passed'] for r in metric_checks):raise ValueError('Compact entry six metrics disagree with validated H/E.')
        for snapshot in state.snapshots:check_frozen(snapshot)
        verify_files(config)
        files=['metrics_combined.csv','image_scores.csv','references.csv','sample_manifest.csv']
        if args.verify_history:files+=['score_checks.csv','metric_checks.csv']
        write_json(output/'run_status.json',dict(complete=True,processed_queries=count,frozen_integrity_passed=True,
            history_parity_passed=True if args.verify_history else None,output_sha256={name:digest(output/name) for name in files}))
    except BaseException as error:
        previous=read_json(output/'run_status.json');write_json(output/'run_status.json',dict(previous,complete=False,error=str(error)));raise
    finally:cleanup()


def main(argv=None):
    args=parse_args(argv)
    if args.status:
        path=project_path(args.status,ROOT);state=read_json(path/'run_status.json');print(path,state)
        if state.get('current_output') and (Path(state['current_output'])/'run_status.json').exists():print(read_json(Path(state['current_output'])/'run_status.json'))
        return
    if args._session:worker(args);return
    if args.resume_from:
        session=project_path(args.resume_from,ROOT);session.relative_to((ROOT/'results'/'test_h').resolve())
        config=read_json(session/'campaign_config.json')
        for k,v in config['options'].items():setattr(args,k,project_path(v,ROOT) if k in ('checkpoint','reference_run') else v)
        verify_files(config);state=read_json(session/'run_status.json')
    else:
        if digest(args.checkpoint)!=WEIGHT_SHA:raise ValueError('Wrong fixed epoch14 checkpoint.')
        files={str(args.checkpoint):WEIGHT_SHA,str(args.checkpoint.parent/'model_manifest.json'):digest(args.checkpoint.parent/'model_manifest.json')}
        for d in args.datasets:
            for s in args.shots:
                inputs=recorded_inputs(args.reference_run,d,s)
                for p in [inputs['meta'],Path(inputs['config']['pretrained_model']),*[inputs['history']/n for n in ('experiment_config.json','input_audit.json','references.csv','sample_manifest.csv','run_status.json')]]:files[str(p)]=digest(p)
                if args.verify_history:
                    for n in ('image_scores.csv','metrics_combined.csv'):files[str(inputs['history']/n)]=digest(inputs['history']/n)
        opts={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_') and k not in ('status','resume_from')}
        config=dict(options=opts,code_sha256=code_hashes(),input_sha256=files,epoch=14,coefficient=.25,local_fraction=.125)
        session=new_directory(ROOT/'results'/'test_h',f'{args.mode}_seed10');write_json(session/'campaign_config.json',config)
        state=dict(complete=False,current=None,current_output=None,completed_jobs=[],job_results={});write_json(session/'run_status.json',state)
    receipt=project_path(args._progress_file,ROOT) if args._progress_file else None
    record_pipeline_progress(receipt,'evaluating',session=str(session),mode=args.mode)
    try:
        for d in args.datasets:
            for s in args.shots:
                key=f'{d}/{s}';previous=state['job_results'].get(key)
                if previous:
                    try:old=read_json(Path(previous)/'run_status.json')
                    except (OSError,ValueError):old={}
                    if old.get('complete'):
                        for n,v in old['output_sha256'].items():
                            if digest(Path(previous)/n)!=v:raise ValueError('Completed output changed.')
                        if key not in state['completed_jobs']:state['completed_jobs'].append(key)
                        continue
                parent=session/d;parent.mkdir(parents=True,exist_ok=True)
                # Reserve an attempt name, but the worker creates the directory.
                output=parent/f'{s}shot_seed10';index=2
                while output.exists():output=parent/f'{s}shot_seed10_run{index:02d}';index+=1
                state.update(complete=False,current=key,current_output=str(output));state['job_results'][key]=str(output);write_json(session/'run_status.json',state)
                command=[sys.executable,str(Path(__file__).resolve()),'--_session',str(session),'--_output',str(output),'--_dataset',d,'--_shot',str(s)]
                for k,v in config['options'].items():
                    if k=='verify_history':command.append('--verify_history' if v else '--no-verify_history')
                    else:command.extend(['--'+k,*map(str,v)] if isinstance(v,list) else ['--'+k,str(v)])
                print(f'{args.mode} {key}: independent process',flush=True);run_child(command)
                if not read_json(output/'run_status.json').get('complete'):raise ValueError('Worker did not finish checks.')
                cleanup();state['completed_jobs'].append(key);write_json(session/'run_status.json',state)
        combined=[]
        for key,folder in state['job_results'].items():
            d,s=key.split('/');combined.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(Path(folder)/'metrics_combined.csv'))
        write_csv(session/'metrics_summary.csv',list(combined[0]),combined)
        verify_files(config);state.update(complete=True,current=None,current_output=None);state.pop('error',None);write_json(session/'run_status.json',state)
        print('Completed:',session)
    except BaseException as error:
        state.update(complete=False,error=str(error));write_json(session/'run_status.json',state)
        print(f"Stopped at {state['current']}; completed={state['completed_jobs']}\nSession={session}\nWorker={state['current_output']}",flush=True);raise


if __name__=='__main__':main()
