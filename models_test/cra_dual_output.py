r"""实验K双输出真实推理验证，不训练：K第9轮CRA图像分数 + H第14轮定位图。

python .\models_test\cra_dual_output.py
六集按数据集→1/2/4-shot、seed10；逐轮独立进程，退出清理后继续。
不使用H定位图最大值重算K图像分数，不改变12.5%/ALIGN/sigma4。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cra_dual_output_utils import CRA_SHA, DUAL, DualBuffer, compare, route
from cra_utils import attach, checked_checkpoint, code_hashes as base_hashes, predict_pair, verify_plan as verify_old_plan
from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, write_csv
from h_inference import WEIGHT_SHA, load_model, recorded_inputs
from rma_align_ablation_utils import DATASETS, verified_history
from rma_control_io import write_json, write_progress
from rma_utils import FIELDS, check_frozen, cleanup, digest, freeze_base, run_child, save_tables

RESULT_ROOT = ROOT/'results'/Path(__file__).stem
FLAGS = ('complete', 'H_K_history_parity_passed', 'dual_donor_identity_passed',
         'offline_mean_parity_passed', 'frozen_integrity_passed')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', default='results/train_cra/mvtec_seed10/best.pth')
    p.add_argument('--k_run', default='results/test_cra/K_seed10')
    p.add_argument('--reference_run', default='results/rma_score_coordination_inference/experiment_H_inference_seed10')
    p.add_argument('--offline_metrics', default='results/test_cra/K_seed10/analysis_20261110/offline_dual_metrics.csv')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--status'); p.add_argument('--resume_from')
    for name in ('session', 'output', 'dataset', 'shot', 'progress_file'):
        p.add_argument('--_'+name, type=int if name == 'shot' else str, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.datasets != sorted(set(args.datasets), key=DATASETS.index) or args.shots != sorted(set(args.shots)) or args.num_workers < 0:
        p.error('Ordered unique datasets/shots and nonnegative workers required.')
    hidden = [getattr(args, '_'+n) is not None for n in ('session', 'output', 'dataset', 'shot')]
    if any(hidden) and not all(hidden): p.error('Incomplete internal worker fields.')
    tokens = sys.argv[1:] if argv is None else argv
    if args.status and args.resume_from: p.error('Use status or resume separately.')
    if args.resume_from and any(t.startswith('--') and t.split('=', 1)[0] != '--resume_from' for t in tokens):
        p.error('Resume restores saved settings; use --resume_from alone.')
    for n in ('checkpoint', 'k_run', 'reference_run', 'offline_metrics'):
        setattr(args, n, project_path(getattr(args, n), ROOT))
    return args


def hashes():
    return dict(base_hashes(), **{str(ROOT/'models_test'/n):digest(ROOT/'models_test'/n)
                for n in ('cra_dual_output.py', 'cra_dual_output_utils.py')})


def verify_files(plan):
    if plan['code_sha256'] != hashes(): raise ValueError('Dual/H/K inference code changed.')
    for path, value in plan['input_sha256'].items():
        if digest(path) != value: raise ValueError('Dual verification input changed: '+path)


def history_inputs(args, dataset, shot):
    from test_cra import verified_complete
    inputs = recorded_inputs(args.reference_run, dataset, shot)
    baseline, _ = verified_history(inputs)
    status = read_json(args.k_run/'run_status.json')
    if status.get('complete') is not True: raise ValueError('Historical K campaign incomplete.')
    folder = Path(status['job_results'][f'{dataset}/{shot}']).resolve(); folder.relative_to(args.k_run)
    verified_complete(folder, len(inputs['manifest']))
    config = read_json(folder/'experiment_config.json')
    if config['cra_checkpoint_sha256'] != CRA_SHA or config['cra_selected_epoch'] != 9:
        raise ValueError('Historical K must be this exact CRA epoch9.')
    if digest(config['h_checkpoint']) != WEIGHT_SHA: raise ValueError('Historical K H底座 changed.')
    for name in ('references.csv', 'sample_manifest.csv'):
        if read_csv(folder/name) != read_csv(inputs['history']/name): raise ValueError('K/H identities differ.')
    offline = [r for r in read_csv(args.offline_metrics) if r['Dataset'] == dataset and int(r['Shot']) == shot]
    if len(offline) != 1 or offline[0]['Condition'] != 'offline_K_image_H_pixel':
        raise ValueError('Missing/duplicate offline dual condition.')
    history = read_csv(folder/'metrics_combined.csv')
    means = {r['Condition']:r for r in history if r['Category'] == 'Mean'}
    for f in FIELDS:
        donor = 'K_CRA' if f.startswith('I-') else 'H'
        if abs(float(offline[0][f])-float(means[donor][f])) > 1e-9:
            raise ValueError('Offline dual metrics do not match their K/H donors.')
    return inputs, folder, history, [r for r in baseline if r['Condition'] == 'baseline'], offline[0]


def preflight(args):
    from test_cra import preflight as k_preflight
    training, selected, sha = checked_checkpoint(args.checkpoint)
    if sha != CRA_SHA or selected['best_epoch'] != 9 or training['h_checkpoint_sha256'] != WEIGHT_SHA:
        raise ValueError('Requires fixed CRA epoch9 and H/new-D epoch14; no retraining or new selection.')
    verify_old_plan(read_json(args.k_run/'campaign_config.json'))
    plan = k_preflight(args)
    plan.update(experiment='K_dual_verification', code_sha256=hashes(),
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                 if not k.startswith('_') and k not in ('status','resume_from')},
        dual_image='K_CRA complete final image score', dual_pixel='H complete final localization map',
        dual_recomputes_image_score=False, mode='real_inference_verification')
    paths = [args.offline_metrics, args.k_run/'run_status.json', args.k_run/'campaign_config.json']
    for d in args.datasets:
        for s in args.shots:
            _, folder, _, _, _ = history_inputs(args, d, s)
            paths += [folder/n for n in ('run_status.json','experiment_config.json','image_scores.csv',
                       'metrics_combined.csv','references.csv','sample_manifest.csv')]
    for path in paths: plan['input_sha256'][str(path)] = digest(path)
    return plan


def restore(args, plan):
    for k,v in plan['options'].items():
        setattr(args,k,project_path(v,ROOT) if k in ('checkpoint','k_run','reference_run','offline_metrics') else v)


def worker(args, session, plan):
    import torch
    from tqdm import tqdm
    key = f'{args._dataset}/{args._shot}'
    if key not in plan['jobs']: raise ValueError('Unscheduled dual worker.')
    output = project_path(args._output,ROOT)
    if output.parent != session/args._dataset: raise ValueError('Dual output escaped dataset directory.')
    output.mkdir(parents=True,exist_ok=False)
    status = dict(complete=False,dataset=args._dataset,shot=args._shot,processed_queries=0)
    write_json(output/'run_status.json',status)
    state = hook = buffer = None
    try:
        verify_files(plan)
        inputs, historical, history, baseline, offline = history_inputs(args,args._dataset,args._shot)
        previous = read_csv(historical/'image_scores.csv')
        write_json(output/'experiment_config.json',dict(inputs['config'],experiment='K_dual_verification',
            h_checkpoint=plan['h_checkpoint'],cra_checkpoint=str(args.checkpoint),cra_epoch=9,
            dual_image=plan['dual_image'],dual_pixel=plan['dual_pixel'],dual_recomputes_image_score=False,
            local_fraction=.125,image_coefficient=.25,result_dir=str(output)))
        for name in ('references.csv','sample_manifest.csv'):
            values=read_csv(inputs['history']/name);write_csv(output/name,list(values[0]),values)
        state=load_model(Path(plan['h_checkpoint']),inputs,output,args.num_workers)
        cra,hook=attach(state,args.checkpoint);cra.eval().requires_grad_(False)
        frozen=freeze_base([None,cra])
        pixels,images,score_rows,score_checks=[],[],[],[]
        closed=set();count=batches=0
        def flush():
            p,i=buffer.finish(state.device);pixels.extend(p);images.extend(i)
            save_tables(output,pixels,images);cleanup()
        with torch.no_grad():
            for items in tqdm(state.loader,desc=f'K dual {args._dataset} {args._shot}-shot'):
                maps,scores,parts=predict_pair(state,hook,items['img'],items['cls_name'],items['img_path'])
                if any(m.shape!=items['img_mask'][:,0].shape for m in maps.values()):
                    raise ValueError('Dual map/ground-truth shape differs.')
                route(maps,scores)
                for i,category in enumerate(items['cls_name']):
                    if count>=len(inputs['manifest']): raise ValueError('Extra dual query.')
                    expected=inputs['manifest'][count];old=previous[count]
                    path,label=items['img_path'][i],int(items['anomaly'][i])
                    if (category!=expected['Category'] or path_key(path)!=path_key(expected['Image Path'])
                            or label!=int(expected['Sample Type']=='Anomaly') or category!=old['Category']
                            or path_key(path)!=path_key(old['Image Path']) or label!=int(old['Label'])):
                        raise ValueError('Dual query order/identity/label differs.')
                    if buffer is None or category!=buffer.category:
                        if buffer is not None: flush();closed.add(buffer.category)
                        if category in closed: raise ValueError('Noncontiguous category.')
                        buffer=DualBuffer(category)
                    buffer.append(items['img_mask'][i,0]>.5,label,maps,scores,i)
                    for name in ('H','K_CRA'):
                        delta=float(scores[name][i])-float(old[name])
                        check=dict(Image=path,Condition=name,Delta=delta,Passed=abs(delta)<=1e-6)
                        score_checks.append(check)
                        if not check['Passed']:
                            write_csv(output/'score_checks.csv',list(check),score_checks)
                            raise ValueError('Actual H/K scores differ from verified K run.')
                    score_rows.append(dict(Category=category,**{'Image Path':path},Label=label,
                        H=float(scores['H'][i]),K_CRA=float(scores['K_CRA'][i]),K_dual=float(scores[DUAL][i]),
                        K_SE=float(parts['K_CRA']['SE'][i]),K_G0=float(parts['K_CRA']['G0'][i])))
                    count+=1
                batches+=1
                status.update(processed_queries=count)
                write_progress(output/'evaluation_progress.json',status)
                del maps,scores,parts
        if buffer is None or count!=plan['jobs'][key]['queries']: raise ValueError('Incomplete dual coverage.')
        flush();table=save_tables(output,pixels,images)
        checks,identity,offline_checks,deltas=compare(table,history,baseline,inputs['classes'],offline)
        for name,values in (('image_scores.csv',score_rows),('score_checks.csv',score_checks),
            ('history_metric_checks.csv',checks),('donor_metric_checks.csv',identity),('offline_checks.csv',offline_checks),
            ('deltas.csv',deltas),('metrics_with_reference.csv',table+baseline)):
            write_csv(output/name,list(values[0]),values)
        if not all(r['Passed'] for r in (*checks,*identity,*offline_checks)):
            raise ValueError('Dual history/donor/offline six-metric verification failed.')
        for snapshot in (*state.snapshots,frozen): check_frozen(snapshot)
        verify_files(plan)
        files=[p for p in output.iterdir() if p.is_file() and p.name not in ('run_status.json','evaluation_progress.json')]
        status.update(**{name:True for name in FLAGS},query_encoder_calls=batches,rma_calls=batches,
            full_current_pixel_identity_checked=True,historical_full_pixel_arrays_compared=False,
            output_sha256={p.name:digest(p) for p in files})
        write_json(output/'run_status.json',status)
    except BaseException as error:
        write_json(output/'run_status.json',dict(status,complete=False,error=str(error) or 'KeyboardInterrupt'));raise
    finally:
        if hook is not None: hook.close()
        state=buffer=None;cleanup()


def verified_complete(folder, queries):
    state=read_json(folder/'run_status.json')
    if not all(state.get(k) is True for k in FLAGS) or state.get('processed_queries')!=queries:
        raise ValueError('Dual worker incomplete or unverified.')
    required={'metrics_combined.csv','image_scores.csv','history_metric_checks.csv','donor_metric_checks.csv',
              'offline_checks.csv','score_checks.csv','deltas.csv','references.csv','sample_manifest.csv'}
    hashes_=state.get('output_sha256',{})
    if not required<=hashes_.keys():raise ValueError('Incomplete dual output receipt.')
    for name,value in hashes_.items():
        if Path(name).name!=name or digest(folder/name)!=value:raise ValueError('Dual output changed: '+name)


def summarize(session,status):
    from rma_matching_control_utils import retention
    metrics=[];deltas=[]
    for key,folder in status['job_results'].items():
        d,s=key.split('/')
        metrics.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(Path(folder)/'metrics_with_reference.csv'))
        deltas.extend(dict(Dataset=d,Shot=int(s),**r) for r in read_csv(Path(folder)/'deltas.csv'))
    for name,values in (('metrics_summary.csv',[r for r in metrics if r['Category']=='Mean']),
        ('metrics_by_category.csv',[r for r in metrics if r['Category']!='Mean']),('deltas_by_category.csv',deltas)):
        write_csv(session/name,list(values[0]),values)
    for comparator in ('H','K_CRA','baseline'):
        values=[r for r in deltas if r['Comparator']==comparator and r['Category']=='Mean']
        write_csv(session/f'deltas_vs_{comparator}.csv',list(values[0]),values)
    records,overall=retention(deltas,comparators=('H','K_CRA','baseline'))
    write_csv(session/'retention_by_shot.csv',list(records[0]),records)
    write_json(session/'verification_summary.json',dict(complete=True,experiment='K_dual_verification',
        actual_H_K_score_parity_passed=True,full_current_H_pixel_identity_passed=True,
        offline_six_metric_parity_passed=True,historical_full_pixel_arrays_compared=False,retention=overall))


def main(argv=None):
    args=parse_args(argv)
    receipt=project_path(args._progress_file,ROOT) if args._progress_file else None
    if args.status:
        session=project_path(args.status,ROOT);session.relative_to(RESULT_ROOT.resolve())
        state=read_json(session/'run_status.json');print('只读最后保存进度，不启动模型，也不判断进程是否仍在运行。')
        print('当前数据集/shot：',state.get('current'));print('已完成：',state.get('completed_jobs',[]));return
    if args._session:
        session=project_path(args._session,ROOT);session.relative_to(RESULT_ROOT.resolve())
        plan=read_json(session/'campaign_config.json');restore(args,plan);worker(args,session,plan);return
    if args.resume_from:
        session=project_path(args.resume_from,ROOT);session.relative_to(RESULT_ROOT.resolve())
        plan=read_json(session/'campaign_config.json');restore(args,plan);verify_files(plan)
        state=read_json(session/'run_status.json')
        if set(state['job_results'])-set(plan['jobs']):raise ValueError('Resume job plan differs.')
    else:
        plan=preflight(args);session=new_directory(RESULT_ROOT,'K_dual_seed10')
        write_json(session/'campaign_config.json',plan)
        state=dict(complete=False,current=None,current_output=None,completed_jobs=[],job_results={})
        write_json(session/'run_status.json',state)
    if receipt:write_json(receipt,dict(session=str(session),phase='testing'))
    try:
        for key,spec in plan['jobs'].items():
            d,s=key.split('/');previous=state['job_results'].get(key)
            state.update(complete=False,current=key,current_output=previous)
            if previous:
                old=Path(previous);old.resolve().relative_to(session)
                try:done=read_json(old/'run_status.json')
                except (OSError,ValueError):done={}
                if done.get('complete'):
                    verified_complete(old,spec['queries'])
                    if key not in state['completed_jobs']:state['completed_jobs'].append(key)
                    continue
            parent=session/d;parent.mkdir(parents=True,exist_ok=True)
            output=parent/f'{s}shot_seed10';attempt=2
            while output.exists():output=parent/f'{s}shot_seed10_run{attempt:02d}';attempt+=1
            state['job_results'][key]=str(output);state.update(current_output=str(output));state.pop('error',None)
            if key in state['completed_jobs']:state['completed_jobs'].remove(key)
            write_json(session/'run_status.json',state)
            print(f'K双输出：数据集={d}，shot={s}',flush=True)
            run_child([sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),
                       '--_output',str(output),'--_dataset',d,'--_shot',s])
            verified_complete(output,spec['queries']);cleanup()
            state['completed_jobs'].append(key);write_json(session/'run_status.json',state)
        summarize(session,state);verify_files(plan)
        state.update(complete=True,current=None,current_output=None);state.pop('error',None)
        write_json(session/'run_status.json',state)
        if receipt:write_json(receipt,dict(session=str(session),phase='completed'))
        print('K双输出验证完成：',session,flush=True)
    except BaseException as error:
        state.update(complete=False,error=str(error) or 'KeyboardInterrupt');write_json(session/'run_status.json',state)
        print(f'运行停止，后续不启动。当前数据集/shot：{state["current"]}\n结果目录：{session}',flush=True);raise


if __name__=='__main__':main()
