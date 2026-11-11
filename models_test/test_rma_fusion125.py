r"""实验L测试：同第14轮主对照＋源域best补充；定位默认12.5%，附25%图。

python .\models_test\test_rma_fusion125.py
自动读取本次训练的epoch_14.pth和best.pth，不强制best为14。
六集各1/2/4-shot；图像评分统一保持H公式，其SE仍用25%局部路径。
每dataset/shot独立子进程，退出清理后再继续；不修改历史结果或正在运行的CRA。
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from fusion_ablation_utils import new_directory,path_key,project_path,read_csv,read_json,write_csv
from rma_control_io import write_json,write_progress
from rma_fusion125_utils import (DATASETS,checked_training,code_hashes,deltas,predict,retention,
                                 target_inputs,target_plan,verify_plan)
from rma_utils import FIELDS,cleanup,digest,run_child

RESULT_ROOT=ROOT/'results'/Path(__file__).stem


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--training_run',default='results/train_rma_fusion125/mvtec_seed10')
    p.add_argument('--reference_run',default='results/rma_score_coordination_inference/experiment_H_inference_seed10')
    p.add_argument('--status');p.add_argument('--resume_from')
    for n in ('session','output','dataset','shot','progress_file'):
        p.add_argument('--_'+n,type=int if n=='shot' else str,help=argparse.SUPPRESS)
    a=p.parse_args(argv);tokens=sys.argv[1:] if argv is None else argv
    if a.status and a.resume_from:p.error('Use status or resume separately.')
    if a.resume_from and any(t.startswith('--') and t.split('=',1)[0]!='--resume_from' for t in tokens):p.error('Resume restores saved options; use alone.')
    hidden=[getattr(a,'_'+n) is not None for n in ('session','output','dataset','shot')]
    if any(hidden) and (not all(hidden) or a.status or a.resume_from):p.error('Invalid internal worker arguments.')
    for n in ('training_run','reference_run'):setattr(a,n,project_path(getattr(a,n),ROOT))
    return a


def preflight(args):
    _,state,weights=checked_training(args.training_run)
    target=target_plan(args.reference_run)
    paths=[args.training_run/n for n in ('campaign_config.json','run_status.json','model_manifest.json','training_history.csv','epoch_14.pth','best.pth')]
    target['input_sha256'].update({str(p):digest(p) for p in paths})
    return dict(target,experiment='L',code_sha256=code_hashes(),weights=weights,
                options=dict(training_run=str(args.training_run),reference_run=str(args.reference_run)),
                primary='epoch14_L125 minus historical H epoch14',secondary=f"best_L125 source-selected epoch{state['best_epoch']} minus H source-best14",
                image_formula='Own-weight H=.75*SE25+.25*G0, unchanged for both localization fractions',
                old_H_L250_source='Historical H image metrics + old G0_L1 pixel metrics; composed, not newly inferred')


def evaluate_weight(inputs,weight,output,progress):
    import torch
    from h_inference import load_model
    from rma_utils import MetricsBuffer,check_frozen,save_tables
    from tqdm import tqdm
    output.mkdir(parents=True,exist_ok=False)
    state=buffer=None;count=0;closed=set();pixel_rows=[];image_rows=[];scores_rows=[]
    try:
        state=load_model(Path(weight['path']),inputs,output,0,expected_sha=weight['sha256'],expected_epoch=weight['epoch'])
        def flush():
            p,i=buffer.finish(state.device);pixel_rows.extend(p);image_rows.extend(i)
            save_tables(output,pixel_rows,image_rows);cleanup()
        with torch.no_grad():
            for batch,items in enumerate(tqdm(state.loader,desc=f"L epoch{weight['epoch']}"),1):
                maps,scores,parts=predict(state,items['img'],items['cls_name'],items['img_path'])
                if any(v.shape!=items['img_mask'][:,0].shape for v in maps.values()):raise ValueError('L map/mask shape differs.')
                if not torch.equal(scores['L125'],scores['L250']):raise ValueError('Localization fraction changed image score.')
                if not torch.allclose(scores['L125'],.75*parts['SE']+.25*parts['G0'],atol=1e-6,rtol=0):raise ValueError('H image formula differs.')
                for i,category in enumerate(items['cls_name']):
                    if count>=len(inputs['manifest']):raise ValueError('Extra L query.')
                    expected=inputs['manifest'][count];path=items['img_path'][i];label=int(items['anomaly'][i])
                    if (category!=expected['Category'] or path_key(path)!=path_key(expected['Image Path']) or label!=int(expected['Sample Type']=='Anomaly')):
                        raise ValueError('L query order/identity differs.')
                    if buffer is None or category!=buffer.category:
                        if buffer is not None:flush();closed.add(buffer.category)
                        if category in closed:raise ValueError('Noncontiguous L category.')
                        buffer=MetricsBuffer(category,conditions=('L125','L250'))
                    buffer.append(items['img_mask'][i,0]>.5,label,maps,scores,i)
                    scores_rows.append(dict(Category=category,**{'Image Path':path},Label=label,
                        H_score=float(scores['L125'][i]),SE25=float(parts['SE'][i]),G0=float(parts['G0'][i])))
                    count+=1
                progress(batch,count);del maps,scores,parts
        if buffer is None or count!=len(inputs['manifest']):raise ValueError('Incomplete L evaluation.')
        flush();table=save_tables(output,pixel_rows,image_rows)
        for category in (*inputs['classes'],'Mean'):
            pair=[r for r in table if r['Category']==category]
            if len(pair)!=2 or any(float(pair[0][f])!=float(pair[1][f]) for f in FIELDS[:3]):
                raise ValueError('Two L localization paths have different image metrics.')
        for snapshot in state.snapshots:check_frozen(snapshot)
        write_csv(output/'image_scores.csv',list(scores_rows[0]),scores_rows)
        return table,scores_rows
    finally:
        state=buffer=None;cleanup()


def worker(args,session,plan):
    from rma_align_ablation_utils import metric_index
    key=f'{args._dataset}/{args._shot}'
    if key not in plan['jobs']:raise ValueError('Unscheduled L worker.')
    output=project_path(args._output,ROOT)
    if output.parent!=session/args._dataset:raise ValueError('L output escaped dataset directory.')
    output.mkdir(parents=True,exist_ok=False)
    status=dict(complete=False,dataset=args._dataset,shot=args._shot,selection=None,processed_queries=0)
    write_json(output/'run_status.json',status)
    try:
        verify_plan(plan);inputs,history,_=target_inputs(args.reference_run,args._dataset,args._shot)
        write_json(output/'experiment_config.json',dict(inputs['config'],experiment='L',weights=plan['weights'],
            train_local_fraction=.125,deploy_local_fractions=[.125,.25],image_formula=plan['image_formula'],
            primary=plan['primary'],secondary=plan['secondary'],result_dir=str(output)))
        for name in ('references.csv','sample_manifest.csv'):
            rows=read_csv(inputs['history']/name);write_csv(output/name,list(rows[0]),rows)
        table=[];cache={}
        for selection,weight in plan['weights'].items():
            status.update(selection=selection,processed_queries=0);write_json(output/'run_status.json',status)
            def progress(batch,count):
                status.update(batch=batch,processed_queries=count);write_progress(output/'evaluation_progress.json',status)
            if weight['sha256'] in cache:
                metrics,score_rows=cache[weight['sha256']]
                status.update(processed_queries=len(inputs['manifest']),same_checkpoint_reused=True)
            else:
                metrics,score_rows=evaluate_weight(inputs,weight,output/selection,progress)
                cache[weight['sha256']]=(metrics,score_rows)
            table.extend(dict(r,Condition=f"{selection}_{r['Condition']}") for r in metrics)
            write_json(output/f'{selection}_checkpoint.json',dict(weight,computed_in_this_selection=(output/selection).exists()))
        metric_index(table,('epoch14_L125','epoch14_L250','best_L125','best_L250'),inputs['classes'])
        difference=deltas(table,history,inputs['classes'])
        for name,rows in (('metrics_combined.csv',table),('metrics_with_reference.csv',table+history),('deltas.csv',difference)):
            write_csv(output/name,list(rows[0]),rows)
        verify_plan(plan)
        files=[p for p in output.rglob('*') if p.is_file() and p.name not in ('run_status.json','evaluation_progress.json')]
        status.update(complete=True,frozen_integrity_passed=True,query_identity_passed=True,
                      image_fraction_invariance_passed=True,output_sha256={str(p.relative_to(output)):digest(p) for p in files})
        write_json(output/'run_status.json',status)
    except BaseException as error:
        write_json(output/'run_status.json',dict(status,complete=False,error=str(error) or 'KeyboardInterrupt'));raise
    finally:cleanup()


def verified_complete(folder,queries):
    state=read_json(folder/'run_status.json')
    if (not all(state.get(k) is True for k in ('complete','frozen_integrity_passed','query_identity_passed','image_fraction_invariance_passed'))
            or state.get('processed_queries')!=queries):raise ValueError('L target round incomplete.')
    if not {'metrics_combined.csv','metrics_with_reference.csv','deltas.csv','references.csv','sample_manifest.csv'}<=state['output_sha256'].keys():
        raise ValueError('Incomplete L output receipt.')
    for name,value in state['output_sha256'].items():
        p=(folder/name).resolve();p.relative_to(folder.resolve())
        if digest(p)!=value:raise ValueError('L target output changed: '+name)


def summarize(session,status):
    metrics=[];changes=[]
    for key,value in status['job_results'].items():
        d,s=key.split('/');folder=Path(value)
        metrics.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(folder/'metrics_with_reference.csv'))
        changes.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(folder/'deltas.csv'))
    for name,rows in (('metrics_summary.csv',[r for r in metrics if r['Category']=='Mean']),
                      ('metrics_by_category.csv',[r for r in metrics if r['Category']!='Mean']),('deltas_by_category.csv',changes)):
        write_csv(session/name,list(rows[0]),rows)
    for comparator in ('H','baseline','old_H_L250'):
        values=[r for r in changes if r['Category']=='Mean' and r['Comparator']==comparator]
        write_csv(session/f'deltas_vs_{comparator}.csv',list(values[0]),values)
    report,overall=retention(changes)
    write_csv(session/'retention_by_shot.csv',list(report[0]),report)
    write_json(session/'verification_summary.json',dict(complete=True,retention=overall,
        primary='epoch14_L125 versus H',secondary='best_L125 versus H',
        old_H_L250_historical_metric_composition=True,no_target_labels_used_for_selection=True))


def main(argv=None):
    args=parse_args(argv)
    if args.status:
        session=project_path(args.status,ROOT);session.relative_to(RESULT_ROOT.resolve())
        state=read_json(session/'run_status.json');print('只读最后保存进度，不启动模型。')
        print('测试：数据集/shot=',state.get('current'),'；全部完成=',state.get('complete'));return
    if args._session:
        session=project_path(args._session,ROOT);session.relative_to(RESULT_ROOT.resolve())
        plan=read_json(session/'campaign_config.json')
        for n,v in plan['options'].items():setattr(args,n,project_path(v,ROOT))
        worker(args,session,plan);return
    if args.resume_from:
        session=project_path(args.resume_from,ROOT);session.relative_to(RESULT_ROOT.resolve())
        plan=read_json(session/'campaign_config.json');verify_plan(plan)
        for n,v in plan['options'].items():setattr(args,n,project_path(v,ROOT))
        status=read_json(session/'run_status.json')
        if set(status['job_results'])-set(plan['jobs']):raise ValueError('Resume L jobs differ.')
    else:
        plan=preflight(args);session=new_directory(RESULT_ROOT,'L_seed10')
        write_json(session/'campaign_config.json',plan)
        status=dict(complete=False,current=None,current_output=None,completed_jobs=[],job_results={})
        write_json(session/'run_status.json',status)
    if args._progress_file:write_json(project_path(args._progress_file,ROOT),dict(session=str(session),phase='testing'))
    try:
        for key,spec in plan['jobs'].items():
            d,s=key.split('/');previous=status['job_results'].get(key)
            status.update(complete=False,current=key,current_output=previous)
            if previous:
                old=Path(previous);old.resolve().relative_to(session)
                if (old/'run_status.json').is_file() and read_json(old/'run_status.json').get('complete'):
                    verified_complete(old,spec['queries'])
                    if key not in status['completed_jobs']:status['completed_jobs'].append(key)
                    continue
            parent=session/d;parent.mkdir(parents=True,exist_ok=True);output=parent/f'{s}shot_seed10';attempt=2
            while output.exists():output=parent/f'{s}shot_seed10_run{attempt:02d}';attempt+=1
            status['job_results'][key]=str(output);status['current_output']=str(output);status.pop('error',None)
            if key in status['completed_jobs']:status['completed_jobs'].remove(key)
            write_json(session/'run_status.json',status)
            print(f'L测试：数据集={d}，shot={s}（第14轮及源域best）',flush=True)
            run_child([sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),
                       '--_output',str(output),'--_dataset',d,'--_shot',s])
            verified_complete(output,spec['queries']);cleanup()
            status['completed_jobs'].append(key);write_json(session/'run_status.json',status)
        summarize(session,status);verify_plan(plan)
        status.update(complete=True,current=None,current_output=None);status.pop('error',None)
        write_json(session/'run_status.json',status);print('L测试完成：',session,flush=True)
    except BaseException as error:
        status.update(complete=False,error=str(error) or 'KeyboardInterrupt');write_json(session/'run_status.json',status)
        print(f"运行停止，后续不启动。测试：数据集/shot={status['current']}\n{session}",flush=True);raise


if __name__=='__main__':main()
