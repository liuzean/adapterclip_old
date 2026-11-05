"""新D同一第14轮权重的RMA全局/局部四条件消融，不训练。

G0_L0全关；G1_L0只开全局项；G0_L1只开局部图；G1_L1完整新D。
默认六数据集各1/2/4-shot，seed10；每dataset/shot独立子进程。
一次查询编码/新D三适配器/RMA计算四条件；ALIGN和平滑始终保留。
"""
import argparse
import logging
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from context_alignment import DATASETS, prepare
from fusion_ablation_utils import compare_baseline, new_directory, path_key, project_path, read_csv, read_json, write_csv, write_json
from new_d_without_rma import reference_checkpoint
from rma_output_ablation_utils import CONDITIONS, FactorBuffer, audit_history, endpoint_metric_checks, route_outputs, save_sample, summarize
from rma_utils import cleanup, digest, failure_status, record_pipeline_progress, run_child, save_tables
from test_rma import preflight


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--shots',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    p.add_argument('--new_d_run',default='results/test_rma_scratch_new/six_datasets_seed10')
    p.add_argument('--off_run',default='results/new_d_without_rma/six_datasets_seed10')
    p.add_argument('--history_session',default='multi_dataset_seed10')
    p.add_argument('--visa_source_root',default='results/branch_diagnostics')
    p.add_argument('--visa_control_root',default='results/dual_output/visa_seed10')
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--save_maps',action='store_true',help='Only selected sample matrices, not the full test set.')
    p.add_argument('--save_visualizations',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--_session',help=argparse.SUPPRESS)
    p.add_argument('--_dataset',choices=DATASETS,help=argparse.SUPPRESS)
    p.add_argument('--_shot',type=int,choices=(1,2,4),help=argparse.SUPPRESS)
    p.add_argument('--_progress_file',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.num_workers<0 or len(set(args.datasets))!=len(args.datasets) or args.shots!=sorted(set(args.shots)):
        p.error('Unique datasets, ascending unique shots, and nonnegative workers required.')
    if Path(args.history_session).name!=args.history_session or args.history_session in ('.','..'):
        p.error('history_session must be a single directory name.')
    worker=[args._session is not None,args._dataset is not None,args._shot is not None]
    if any(worker) and not all(worker):
        p.error('All worker fields required together.')
    for key in ('new_d_run','off_run','visa_source_root','visa_control_root'):
        setattr(args,key,project_path(getattr(args,key),ROOT))
    return args


def worker(args):
    import torch
    from fusion_ablation import infer_batch, initialize
    from rma_joint_utils import clone_adapters, encode_learned_text, load_joint_adapters
    from rma_model import ReferenceMatchingAdapter
    from rma_scratch_common import validate_scratch
    from rma_utils import check_frozen, freeze_base
    from test import prompt_association
    from tools import setup_seed
    from tqdm import tqdm
    session=project_path(args._session,ROOT)
    session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    output=session/args._dataset/f'{args._shot}shot_seed10';output.mkdir(parents=True,exist_ok=False)
    logger=logging.getLogger('rma-output-ablation');logger.setLevel(logging.INFO)
    handler=logging.FileHandler(output/'test.log',encoding='utf-8');logger.addHandler(handler)
    try:
        source,control,audit=prepare(args,args._dataset,args._shot)
        full=args.new_d_run/args._dataset/f'{args._shot}shot_seed10'
        off=args.off_run/args._dataset/f'{args._shot}shot_seed10'
        audit_history(source,audit,full,off)
        if audit!=read_json(session/'input_audit.json')[args._dataset][str(args._shot)]:
            raise ValueError('Inputs changed since preflight.')
        checkpoint,checkpoint_hash=reference_checkpoint(args)
        if checkpoint_hash!=read_json(session/'campaign_config.json')['checkpoint_sha256']:
            raise ValueError('Checkpoint changed since preflight.')
        write_json(output/'experiment_config.json',dict(source['config'],new_d_checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hash,weight_epoch=14,conditions=list(CONDITIONS),align_used=True,
            num_workers=args.num_workers,save_maps=args.save_maps,save_visualizations=args.save_visualizations))
        write_json(output/'input_audit.json',audit)
        write_csv(output/'references.csv',['Category','Image Path'],[{'Category':c,'Image Path':p} for c,p in source['reference_paths']])
        write_csv(output/'sample_manifest.csv',list(source['manifest'][0]),source['manifest'])
        write_json(output/'run_status.json',dict(status='initializing',complete=False))
        setup_seed(10)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        validate_scratch(payload,digest(source['config']['checkpoint_path']),digest(source['config']['pretrained_model']),source['config'],'new')
        if payload['epoch']!=14:
            raise ValueError('Requires new-D epoch14.')
        initialized=initialize(source,output,args.num_workers,logger)
        config,model,textual,visual,pq,text,memory,loader,dpam,device=initialized
        original_snapshot=freeze_base(initialized)
        adapters=clone_adapters(initialized);load_joint_adapters(adapters,payload)
        rma=ReferenceMatchingAdapter(**payload['model_spec']).to(device).eval()
        rma.load_state_dict(payload['rma'],strict=True);rma.requires_grad_(False);del payload
        candidate_snapshot=freeze_base([None,*adapters,rma])
        with torch.no_grad():
            candidate_text=encode_learned_text(model,adapters[0])
        historical={'G1_L1':read_csv(full/'image_scores.csv'),'G0_L0':read_csv(off/'image_scores.csv')}
        selected={path_key(r['Image Path']):r for r in source['diagnostics']}
        scores_saved=[];score_checks=[];map_checks=[];pixels=[];images=[]
        count=batches=0;buffer=None;closed=set();visited=set()

        def persist():
            for file,values in [('image_scores.csv',scores_saved),('endpoint_score_checks.csv',score_checks),('baseline_map_checks.csv',map_checks)]:
                if values:
                    write_csv(output/file,list(values[0]),values)

        def flush():
            p,i=buffer.finish(device);pixels.extend(p);images.extend(i)
            save_tables(output,pixels,images);persist();cleanup()

        with torch.no_grad():
            for items in tqdm(loader,desc=f'{args._dataset} {args._shot}-shot / RMA G,L four conditions'):
                paths=items['img_path'];classes=items['cls_name']
                features,patches=model.encode_image(items['img'].to(device),config.features_list,DPAM_layer=dpam)
                pf,pp=prompt_association(*memory,classes)
                a_raw,a_maps,a_scores,_,a_native,a_fixed,_=infer_batch(config,features,patches,pf,pp,textual,visual,pq,text,paths)
                versions=[v._version for v in (features,pf,*patches,*pp)]
                old,maps0,scores0,_,_,fixed,extra=infer_batch(config,features,patches,pf,pp,*adapters,candidate_text,paths)
                g,l=rma(patches,pp)
                raw,maps,scores,components=route_outputs(old,maps0,scores0,fixed,extra,g,l,config.sigma)
                if versions!=[v._version for v in (features,pf,*patches,*pp)]:
                    raise ValueError('Candidate modified shared CLIP inputs.')
                raw['baseline']=a_raw['baseline'];maps['baseline']=a_maps['baseline'];scores['baseline']=a_scores['baseline']
                for n in ('baseline',*CONDITIONS):
                    if maps[n].shape!=items['img_mask'][:,0].shape or scores[n].shape!=(len(paths),):
                        raise ValueError('Unexpected map/score shape.')
                    if not all(torch.isfinite(v).all() for v in (raw[n],maps[n],scores[n])):
                        raise ValueError('Nonfinite condition output.')
                masks=(items['img_mask'][:,0]>.5).to(torch.uint8)
                for i,category in enumerate(classes):
                    expected=source['manifest'][count];label=int(items['anomaly'][i])
                    if (category!=expected['Category'] or path_key(paths[i])!=path_key(expected['Image Path'])
                            or label!=int(expected['Sample Type']=='Anomaly')):
                        raise ValueError('Query identity/order/label mismatch.')
                    if buffer is None or category!=buffer.category:
                        if buffer is not None:
                            flush();closed.add(buffer.category)
                        if category in closed:
                            raise ValueError('Noncontiguous categories.')
                        buffer=FactorBuffer(category)
                    buffer.append(masks[i],label,maps,scores,i)
                    row={'Category':category,'Image Path':paths[i],'Label':label,
                         **{f'{n} Image Score':float(scores[n][i]) for n in ('baseline',*CONDITIONS)},
                         **{n:float(v[i]) for n,v in components.items()}}
                    scores_saved.append(row)
                    previous=control['scores'][count]
                    baseline_ok=math.isclose(float(a_scores['baseline'][i]),float(previous['baseline Image Score']),abs_tol=1e-6,rel_tol=1e-5)
                    baseline_ok=baseline_ok and all(math.isclose(float(a_native[n][i]),float(previous[f'{n} Global']),abs_tol=1e-6,rel_tol=1e-5) for n in ('VA','TA','PQA'))
                    baseline_ok=baseline_ok and math.isclose(float(a_fixed[i]),float(previous['Fixed Global Mean']),abs_tol=1e-6,rel_tol=1e-5)
                    score_checks.append(dict(Image=paths[i],Condition='baseline',Passed=baseline_ok))
                    for name,column in [('G1_L1','S_new Image Score'),('G0_L0','new_D_without_RMA Image Score')]:
                        h=historical[name][count]
                        same=(category==h['Category'] and path_key(paths[i])==path_key(h['Image Path']) and label==int(h['Label']))
                        same=same and math.isclose(row[f'{name} Image Score'],float(h[column]),abs_tol=1e-6,rel_tol=1e-5)
                        score_checks.append(dict(Image=paths[i],Condition=name,Passed=same))
                    if not all(r['Passed'] for r in score_checks[-3:]):
                        persist();raise ValueError('Historical original-A/all-on/all-off per-image parity failed.')
                    key=path_key(paths[i])
                    if key in selected:
                        map_checks.extend(save_sample(args,source,output,selected[key],items['img'][i],masks[i],raw,maps,i))
                        visited.add(key)
                        if not all(r['Passed'] for r in map_checks):
                            persist();raise ValueError('Original-A selected map parity failed.')
                    count+=1
                batches+=1
                del features,patches,pf,pp,a_raw,a_maps,a_scores,old,maps0,scores0,g,l,raw,maps,scores,extra
        if buffer is None or count!=len(source['manifest']) or visited!=set(selected):
            raise ValueError('Incomplete test coverage.')
        flush();table=save_tables(output,pixels,images)
        checks=compare_baseline(source,table,.001)
        write_csv(output/'baseline_reference_comparison.csv',list(checks[0]),checks)
        if not all(r['Passed'] for r in checks):
            raise ValueError('Original-A metric parity failed.')
        endpoint_metric_checks(output,table,full,off)
        check_frozen(original_snapshot);check_frozen(candidate_snapshot)
        if digest(checkpoint)!=checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation.')
        write_json(output/'run_status.json',dict(status='completed',complete=True,sample_count=count,
            query_encoder_calls=batches,new_D_three_adapter_calls=batches,rma_calls=batches,
            baseline_parity_passed=True,endpoint_parity_passed=True,frozen_integrity_passed=True,
            weight_epoch=14,conditions=list(CONDITIONS),align_used=True))
    except BaseException as error:
        failure_status(output,error);logger.exception('RMA output ablation stopped.');raise
    finally:
        cleanup();handler.close();logger.removeHandler(handler)


def main(argv=None):
    args=parse_args(argv)
    if args._session:
        worker(args);return
    checkpoint,checkpoint_hash=reference_checkpoint(args)
    off_campaign=read_json(args.off_run/'campaign_config.json')
    if off_campaign.get('checkpoint_sha256')!=checkpoint_hash or read_json(args.off_run/'run_status.json').get('complete') is not True:
        raise ValueError('All-off history must use the identical completed new-D checkpoint.')
    jobs,audit=preflight(args)
    for dataset,shot in jobs:
        source,_,one=prepare(args,dataset,shot)
        audit_history(source,one,args.new_d_run/dataset/f'{shot}shot_seed10',args.off_run/dataset/f'{shot}shot_seed10')
    session=new_directory(ROOT/'results'/Path(__file__).stem,'six_datasets_seed10')
    receipt=project_path(args._progress_file,ROOT) if args._progress_file else None
    record_pipeline_progress(receipt,'testing',session=str(session),evaluation_jobs=jobs)
    write_json(session/'input_audit.json',audit)
    write_json(session/'campaign_config.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        datasets=args.datasets,shots=args.shots,seed=10,weight_epoch=14,conditions=list(CONDITIONS),
        new_d_run=str(args.new_d_run),off_run=str(args.off_run),align_used=True,
        definition='G controls native global mean only; L controls local map before ALIGN. Image scores always include own smoothed map max.'))
    done=[]
    try:
        for dataset,shot in jobs:
            write_json(session/'run_status.json',dict(status='testing',complete=False,current=[dataset,shot],completed_jobs=done))
            print(f'[{len(done)+1}/{len(jobs)}] {dataset} {shot}-shot / RMA全局局部四条件',flush=True)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),'--_dataset',dataset,'--_shot',str(shot)]
            for name in ('new_d_run','off_run','history_session','visa_source_root','visa_control_root','num_workers'):
                command.extend([f'--{name}',str(getattr(args,name))])
            if args.save_maps:
                command.append('--save_maps')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            run_child(command)
            state=read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json')
            if not all(state.get(k) is True for k in ('complete','baseline_parity_passed','endpoint_parity_passed','frozen_integrity_passed')):
                raise ValueError('Incomplete or failed round checks.')
            done.append((dataset,shot));summarize(session,done,DATASETS);cleanup()
        write_json(session/'run_status.json',dict(status='completed',complete=True,completed_jobs=done))
        record_pipeline_progress(receipt,'completed',session=str(session),evaluation_jobs=jobs)
    except BaseException as error:
        failure_status(session,error)
        print(f'停止：{dataset} {shot}-shot，已完成{len(done)}/{len(jobs)}；结果：{session}',flush=True)
        raise
    print(f'Completed: {session}',flush=True)


if __name__=='__main__':
    main()
