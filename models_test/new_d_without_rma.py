"""新D第14轮同权重关闭RMA；直接调用原test.test，不复制原推理循环。

默认六数据集各1/2/4-shot、seed10。逐条件独立子进程，自动核对原采样器实际
选出的参考图及查询身份。仅加载三个适配器；不创建RMA、不重训、不修改test.py。
"""
import argparse
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from context_alignment import DATASETS, prepare
from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, write_csv, write_json
from rma_utils import FIELDS, cleanup, digest, failure_status, record_pipeline_progress, run_child, save_tables
from test_rma import preflight
from test_rma_joint import summarize

CANDIDATE = 'new_D_without_RMA'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--shots',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    parser.add_argument('--new_d_run',default='results/test_rma_scratch_new/six_datasets_seed10')
    parser.add_argument('--history_session',default='multi_dataset_seed10')
    parser.add_argument('--visa_source_root',default='results/branch_diagnostics')
    parser.add_argument('--visa_control_root',default='results/dual_output/visa_seed10')
    parser.add_argument('--_session',help=argparse.SUPPRESS)
    parser.add_argument('--_dataset',choices=DATASETS,help=argparse.SUPPRESS)
    parser.add_argument('--_shot',type=int,choices=(1,2,4),help=argparse.SUPPRESS)
    parser.add_argument('--_progress_file',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if len(set(args.datasets))!=len(args.datasets) or args.shots!=sorted(set(args.shots)):
        parser.error('Unique datasets and ascending unique shots required.')
    if Path(args.history_session).name!=args.history_session or args.history_session in ('.','..'):
        parser.error('history_session must be a single directory name.')
    worker=[args._session is not None,args._dataset is not None,args._shot is not None]
    if any(worker) and not all(worker):
        parser.error('All internal worker fields are required together.')
    for name in ('new_d_run','visa_source_root','visa_control_root'):
        setattr(args,name,project_path(getattr(args,name),ROOT))
    return args


def reference_checkpoint(args):
    """Bind ablation to the exact completed full new-D evaluation checkpoint."""
    campaign=read_json(args.new_d_run/'campaign_config.json')
    checkpoint=project_path(campaign['checkpoint'],ROOT)
    state=read_json(checkpoint.parent/'run_status.json')
    manifest=read_json(checkpoint.parent/'model_manifest.json')
    if (read_json(args.new_d_run/'run_status.json').get('complete') is not True
            or campaign.get('candidate')!='S_new' or campaign.get('flow')!='new'
            or state.get('complete') is not True or state.get('best_epoch')!=14
            or manifest.get('format')!='adaptclip-rma-scratch-v1' or manifest.get('flow')!='new'
            or checkpoint.resolve()!=Path(state['selected_checkpoint']).resolve()
            or digest(checkpoint)!=campaign['checkpoint_sha256']):
        raise ValueError('Requires the exact source-selected epoch14 best.pth used by completed full new D.')
    return checkpoint,campaign['checkpoint_sha256']


def check_identities(actual, expected, label):
    """Preserve class, reference order and multiplicity, not just path sets."""
    checks=[]
    if len(actual)!=len(expected):
        raise ValueError(f'{label} count differs: actual {len(actual)}, expected {len(expected)}')
    for index,(a,b) in enumerate(zip(actual,expected)):
        passed=a[0]==b[0] and path_key(a[1])==path_key(b[1]) and a[2:]==b[2:]
        checks.append(dict(Index=index,Category=a[0],Actual=a[1],Historical=b[1],Passed=passed))
    return checks


def historical_comparison(args,session,jobs):
    deltas=[]
    for dataset,shot in jobs:
        name=Path(dataset)/f'{shot}shot_seed10'/'metrics_combined.csv'
        current=next(r for r in read_csv(session/name) if r['Category']=='Mean' and r['Condition']==CANDIDATE)
        full=next(r for r in read_csv(args.new_d_run/name) if r['Category']=='Mean' and r['Condition']=='S_new')
        change={f:float(current[f])-float(full[f]) for f in FIELDS}
        deltas.append(dict(Dataset=dataset,Shot=shot,Condition=CANDIDATE,Comparator='S_new',**change,Net=sum(change.values())))
    write_csv(session/'deltas_vs_new_D.csv',list(deltas[0]),deltas)


def worker(args):
    import torch
    import test as original
    from rma_scratch_common import validate_scratch
    from tools import setup_seed
    session=project_path(args._session,ROOT)
    session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    output=session/args._dataset/f'{args._shot}shot_seed10'
    output.mkdir(parents=True,exist_ok=False)
    checkpoint,checkpoint_hash=reference_checkpoint(args)
    campaign=read_json(session/'campaign_config.json')
    if checkpoint_hash!=campaign['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since campaign preflight.')
    source,_,audit=prepare(args,args._dataset,args._shot)
    if audit!=read_json(session/'input_audit.json')[args._dataset][str(args._shot)]:
        raise ValueError('Historical input changed after preflight.')
    original_prompt,original_dataset,original_evaluator=original.PromptDataset,original.Dataset,original.Evaluator
    identities=dict(references=False,queries=False,scores=False)
    category_metrics=[]
    runtime=SimpleNamespace(**source['config'])
    runtime.checkpoint_path=str(checkpoint)
    runtime.save_path=str(output/'original_test')
    runtime.eval_metrics=list(FIELDS)
    runtime.class_name=None

    def guarded_prompt(*values,**settings):
        data=original_prompt(*values,**settings)
        actual=[(r['cls_name'],str(Path(data.root)/r['img_path'])) for r in data.prompt_data_all]
        expected=list(source['reference_paths'])
        checks=check_identities(actual,expected,'Reference')
        write_csv(output/'reference_identity_checks.csv',list(checks[0]),checks)
        write_csv(output/'references.csv',['Category','Image Path'],
                  [{'Category':c,'Image Path':p} for c,p in actual])
        if not all(r['Passed'] for r in checks):
            raise ValueError('Original PromptDataset sampled different references/order; evaluation stopped.')
        identities['references']=True
        return data

    def guarded_queries(*values,**settings):
        data=original_dataset(*values,**settings)
        actual=[(r['cls_name'],str(Path(data.root)/r['img_path']),int(r['anomaly'])) for r in data.data_all]
        expected=[(r['Category'],r['Image Path'],int(r['Sample Type']=='Anomaly')) for r in source['manifest']]
        checks=check_identities(actual,expected,'Query')
        write_csv(output/'query_identity_checks.csv',list(checks[0]),checks)
        if not all(r['Passed'] for r in checks):
            raise ValueError('Original Dataset queries/order/labels differ from history.')
        identities['queries']=True
        return data

    class RecordingEvaluator(original_evaluator):
        def run(self,results,cls_name,logger=None):
            # Read the actual tensors reaching the unchanged original evaluator.
            if not identities['scores']:
                labels=results['gt_anomalys'].detach().cpu().tolist()
                scores=results['pr_anomalys'].detach().cpu().tolist()
                actual=[(str(c),str(p),int(y)) for c,p,y in zip(results['cls_names'],results['query_paths'],labels)]
                expected=[(r['Category'],r['Image Path'],int(r['Sample Type']=='Anomaly')) for r in source['manifest']]
                checks=check_identities(actual,expected,'Evaluated query')
                if len(scores)!=len(expected) or not all(r['Passed'] for r in checks):
                    raise ValueError('Final scored queries differ from historical identities.')
                if not all(math.isfinite(float(s)) for s in scores):
                    raise ValueError('Nonfinite final image scores.')
                records=[{'Category':c,'Image Path':p,'Label':y,f'{CANDIDATE} Image Score':float(s)}
                         for (c,p,y),s in zip(actual,scores)]
                write_csv(output/'image_scores.csv',list(records[0]),records)
                identities['scores']=True
            values=super().run(results,cls_name,logger)
            record=dict(Category=cls_name,Condition=CANDIDATE,**{f:float(values[f])*100 for f in FIELDS})
            if not all(math.isfinite(record[f]) and 0<=record[f]<=100 for f in FIELDS):
                raise ValueError('Invalid candidate metric.')
            category_metrics.append(record)
            return values

    try:
        write_json(output/'experiment_config.json',dict(vars(runtime),seed=10,
            checkpoint_sha256=checkpoint_hash,weight_epoch=14,rma_used=False,align_used=True,
            baseline_source='Historical original A, not a fresh baseline inference',
            inference_implementation='Unmodified test.test; dataset constructors audited; evaluator delegates to original'))
        write_json(output/'input_audit.json',audit)
        write_csv(output/'sample_manifest.csv',list(source['manifest'][0]),source['manifest'])
        write_json(output/'run_status.json',dict(status='running',complete=False))
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        validate_scratch(payload,digest(source['config']['checkpoint_path']),
                         digest(source['config']['pretrained_model']),source['config'],'new')
        if payload['epoch']!=14:
            raise ValueError('Checkpoint payload is not epoch14.')
        del payload
        original.PromptDataset,original.Dataset,original.Evaluator=guarded_prompt,guarded_queries,RecordingEvaluator
        setup_seed(10)
        original.test(runtime)
        if not all(identities.values()) or {r['Category'] for r in category_metrics}!=set(source['classes']):
            raise ValueError('Incomplete audited test output.')
        if len(category_metrics)!=len(source['classes']):
            raise ValueError('Duplicate category evaluation.')
        historical=[dict(Category=c,Condition='baseline',**{f:source['metrics'][(c,f)] for f in FIELDS})
                    for c in source['classes']]
        pixels,images=[],[]
        for record in historical+category_metrics:
            for level,destination in [('P',pixels),('I',images)]:
                destination.append(dict(Category=record['Category'],Condition=record['Condition'],
                    **{m:record[f'{level}-{m}'] for m in ('AUROC','AP','F1max')}))
        save_tables(output,pixels,images)
        if digest(checkpoint)!=checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation.')
        write_json(output/'run_status.json',dict(status='completed',complete=True,
            sample_count=len(source['manifest']),reference_identity_passed=True,query_identity_passed=True,
            candidate=CANDIDATE,weight_epoch=14,rma_used=False,align_used=True,
            original_test_executed=True,historical_baseline_only=True))
    except BaseException as error:
        failure_status(output,error)
        raise
    finally:
        original.PromptDataset,original.Dataset,original.Evaluator=original_prompt,original_dataset,original_evaluator
        # Close only the logger handlers opened for this subprocess before exit.
        for value in logging.Logger.manager.loggerDict.values():
            if isinstance(value,logging.Logger):
                for handler in value.handlers[:]:
                    handler.close();value.removeHandler(handler)
        cleanup()


def main(argv=None):
    args=parse_args(argv)
    if args._session:
        worker(args)
        return
    checkpoint,checkpoint_hash=reference_checkpoint(args)
    jobs,audit=preflight(args)
    for dataset,shot in jobs:
        previous=args.new_d_run/dataset/f'{shot}shot_seed10'
        if (read_json(previous/'run_status.json').get('complete') is not True
                or read_json(previous/'input_audit.json')!=audit[dataset][str(shot)]):
            raise ValueError('Historical full new-D round/inputs do not match the ablation.')
    session=new_directory(ROOT/'results'/Path(__file__).stem,'six_datasets_seed10')
    receipt=project_path(args._progress_file,ROOT) if args._progress_file else None
    record_pipeline_progress(receipt,'testing',session=str(session),evaluation_jobs=jobs)
    write_json(session/'input_audit.json',audit)
    write_json(session/'campaign_config.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        weight_epoch=14,datasets=args.datasets,shots=args.shots,seed=10,new_d_run=str(args.new_d_run),
        candidate=CANDIDATE,rma_used=False,align_used=True,
        baseline_source='Historical original A; no fresh original-A rerun'))
    done=[]
    try:
        for dataset,shot in jobs:
            write_json(session/'run_status.json',dict(status='testing',complete=False,current=[dataset,shot],completed_jobs=done))
            print(f'[{len(done)+1}/{len(jobs)}] 新D同权重关闭RMA：{dataset} {shot}-shot',flush=True)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),
                     '--_dataset',dataset,'--_shot',str(shot)]
            for name in ('new_d_run','history_session','visa_source_root','visa_control_root'):
                command.extend([f'--{name}',str(getattr(args,name))])
            run_child(command)
            state=read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json')
            if not all(state.get(k) is True for k in ('complete','reference_identity_passed','query_identity_passed')):
                raise ValueError('Child did not confirm complete audited evaluation.')
            done.append((dataset,shot))
            summarize(session,done,CANDIDATE)
            historical_comparison(args,session,done)
            cleanup()
        write_json(session/'run_status.json',dict(status='completed',complete=True,completed_jobs=done))
        record_pipeline_progress(receipt,'completed',session=str(session),evaluation_jobs=jobs)
    except BaseException as error:
        failure_status(session,error)
        print(f'运行停止：{dataset} {shot}-shot；已完成{len(done)}/{len(jobs)}；结果：{session}',flush=True)
        raise
    print(f'Completed: {session}',flush=True)


if __name__=='__main__':
    main()
