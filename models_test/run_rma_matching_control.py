r"""实验I：先训练cosine对照，再按固定H公式测试六集各1/2/4-shot。

python .\models_test\run_rma_matching_control.py
--status只读；--resume_training_from从完整epoch恢复训练，再自动测试。
--resume_from复用本序列完整训练并继续测试。
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import new_directory,project_path,read_json
from h_inference import recorded_inputs
from rma_matching_control_utils import check_control_settings,control_checkpoint,verified_history
from rma_control_io import write_json
from rma_utils import cleanup,digest,run_child
from run_rma_experiment_sequence import completed_session


def inspect_progress(session,state):
    """Read persisted progress even if a child file is absent or damaged."""
    result=dict(stage=state.get('current_stage'),read_errors=[])
    def safe_read(path):
        if not path.is_file():return {}
        try:
            value=read_json(path)
            if not isinstance(value,dict):raise ValueError('Expected JSON object')
            return value
        except (OSError,ValueError) as error:
            result['read_errors'].append(f'{path}: {error}');return {}
    report=safe_read(session/f"{state.get('current_stage')}_progress.json")
    if report.get('session'):
        folder=Path(report['session']);result['child_session']=str(folder)
        child=safe_read(folder/'run_status.json');result['child_status']=child
        if state.get('current_stage')=='train':
            result['training_progress']=safe_read(folder/'training_progress.json')
        elif child.get('current_output'):
            result['worker_status']=safe_read(Path(child['current_output'])/'run_status.json')
    return result


def stop_position(state,snapshot):
    """Keep user-facing stop records to only the requested location fields."""
    stage=state.get('current_stage');child=snapshot.get('child_status',{})
    if stage=='train':
        progress=snapshot.get('training_progress',{})
        return dict(stage='train',epoch=progress.get('epoch'),batch=progress.get('batch'))
    worker=snapshot.get('worker_status',{})
    current=child.get('current') or ''
    dataset,_,shot=current.partition('/')
    return dict(stage='test',dataset=dataset or worker.get('dataset'),shot=shot or worker.get('shot'))


def describe(session,state,snapshot=None):
    if state.get('complete'):return '训练和测试均已完成。'
    snapshot=inspect_progress(session,state) if snapshot is None else snapshot
    position=stop_position(state,snapshot)
    if position['stage']=='train':
        return f"训练：epoch={position['epoch'] if position['epoch'] is not None else '尚无记录'}，batch={position['batch'] if position['batch'] is not None else '尚无记录'}"
    return f"测试：数据集={position['dataset'] or '尚无记录'}，shot={position['shot'] or '尚无记录'}"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--status');p.add_argument('--resume_from');p.add_argument('--resume_training_from')
    a=p.parse_args(argv)
    if sum(bool(x) for x in (a.status,a.resume_from,a.resume_training_from))>1:
        p.error('Use status, resume_from or resume_training_from separately.')
    if a.status:
        session=project_path(a.status,ROOT);s=read_json(session/'run_status.json')
        print('全部完成。' if s.get('complete') else '以下只显示最后保存进度，不启动模型，也不据此判断进程是否仍在运行。')
        print(describe(session,s));return
    reference=ROOT/'results/rma_score_coordination_inference/experiment_H_inference_seed10'
    try:
        from rma_scratch_common import parse_training
        settings=parse_training('new',[],matching_control=True);check_control_settings(settings)
        if a.resume_training_from:
            from rma_control_resume import inspect_source
            a.resume_training_from=project_path(a.resume_training_from,ROOT)
            inspect_source(a.resume_training_from,settings)
        # Check all target ledgers and comparison fingerprints before training.
        for d in DATASETS:
            for shot in (1,2,4):verified_history(recorded_inputs(reference,d,shot))
    except BaseException:
        print('停止于启动前检查；没有启动本次训练或目标测试。',file=sys.stderr,flush=True)
        raise
    entry=ROOT/'models_test'
    names=('run_rma_matching_control.py','train_rma_cosine.py','test_rma_cosine.py','rma_matching_control_utils.py',
           'rma_model.py','rma_scratch_common.py','h_inference.py','rma_control_io.py','rma_control_resume.py')
    hashes={name:digest(entry/name) for name in names}
    if a.resume_from:
        session=project_path(a.resume_from,ROOT);session.relative_to((ROOT/'results/run_rma_matching_control').resolve())
        config=read_json(session/'execution_plan.json');state=read_json(session/'run_status.json')
        if config['code_sha256']!=hashes:raise ValueError('Sequence code changed; cannot silently resume.')
        if state.get('complete'):print('实验I已完成：',session);return
        training=state['stage_results'].get('train')
        if training is None:
            receipt=session/'train_progress.json'
            if receipt.is_file():training=read_json(receipt).get('session')
        if not training or not (Path(training)/'run_status.json').is_file() or read_json(Path(training)/'run_status.json').get('complete') is not True:
            raise ValueError('Training incomplete: use --resume_training_from with the actual training directory to recover at an epoch boundary.')
        selected=Path(training)/'best.pth';control_checkpoint(selected)
        state['stage_results']['train']=str(Path(training));state['completed_stages']=['train']
    else:
        session=new_directory(ROOT/'results/run_rma_matching_control','I_cosine_seed10')
        config=dict(experiment='I',stages=['train','test'],code_sha256=hashes,seed=10,datasets=list(DATASETS),shots=[1,2,4],
                    selection='New-D source rule; independent best, not forced epoch14',reference=str(reference),
                    resume_training_from=str(a.resume_training_from) if a.resume_training_from else None,
                    cleanup='Each stage/target round in subprocess; wait exit before next; training cleanup each epoch.')
        write_json(session/'execution_plan.json',config)
        state=dict(complete=False,current_stage='train',completed_stages=[],stage_results={})
        write_json(session/'run_status.json',state)
    try:
        if 'train' not in state['completed_stages']:
            state.update(current_stage='train',complete=False);write_json(session/'run_status.json',state)
            print('[1/2] 实验I：同Top8，仅余弦加权，从头训练四适配器',flush=True)
            receipt=session/'train_progress.json'
            command=[sys.executable,'-u',str(entry/'train_rma_cosine.py'),'--_progress_file',str(receipt)]
            if a.resume_training_from:command.extend(['--resume_training_from',str(a.resume_training_from)])
            run_child(command)
            cleanup();training=completed_session(receipt,'train_rma_cosine')
            control_checkpoint(training/'best.pth');state['stage_results']['train']=str(training);state['completed_stages'].append('train')
        selected=Path(state['stage_results']['train'])/'best.pth'
        _,train_status,weight_hash=control_checkpoint(selected)
        write_json(session/'selected_checkpoint.json',dict(checkpoint=str(selected),sha256=weight_hash,epoch=train_status['best_epoch']))
        state.update(current_stage='test',complete=False);write_json(session/'run_status.json',state)
        print(f"[2/2] 固定H公式，测试六集1/2/4-shot；本次源域best epoch={train_status['best_epoch']}",flush=True)
        receipt=session/'test_progress.json'
        previous=read_json(receipt).get('session') if a.resume_from and receipt.is_file() else None
        if previous:
            cmd=[sys.executable,'-u',str(entry/'test_rma_cosine.py'),'--resume_from',previous]
        else:cmd=[sys.executable,'-u',str(entry/'test_rma_cosine.py'),'--checkpoint',str(selected),'--_progress_file',str(receipt)]
        run_child(cmd);cleanup();testing=completed_session(receipt,'test_rma_cosine')
        state['stage_results']['test']=str(testing);state['completed_stages'].append('test')
        state.update(complete=True,current_stage=None);state.pop('error',None);write_json(session/'run_status.json',state)
        print('实验I训练与18条件测试全部完成：',session,flush=True)
    except BaseException as error:
        state.update(complete=False,error=str(error) or 'Ctrl+C')
        snapshot=inspect_progress(session,state)
        position=stop_position(state,snapshot)
        summary='运行停止，后续不启动。\n'+describe(session,state,snapshot)
        print(summary,file=sys.stderr,flush=True)
        try:
            # Keep a separate record for each stop, plus a latest summary.
            stop=new_directory(session/'stop_records','stop')
            write_json(stop/'progress.json',position)
            (stop/'summary.txt').write_text(summary+'\n',encoding='utf-8')
            (session/'stop_summary.txt').write_text(summary+'\n',encoding='utf-8')
            state.update(stop_progress=position,last_stop_record=str(stop))
            write_json(session/'run_status.json',state)
        except OSError as save_error:
            print(f'停止摘要保存失败：{save_error}',file=sys.stderr,flush=True)
        raise
    finally:cleanup()


if __name__=='__main__':main()
