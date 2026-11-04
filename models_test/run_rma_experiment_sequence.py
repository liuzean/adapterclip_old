"""依次运行：旧B双输出(save_maps) → 原流程训练/测试 → 新流程训练/测试。

默认六数据集、各1/2/4-shot、seed10。每阶段独立子进程，等退出后继续。
训练每epoch调用已有cleanup；测试每dataset/shot使用独立子进程释放资源。
父调度器不导入torch、不加载模型；失败/中断停止，不自动重跑或覆盖结果。
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json, write_json
from rma_utils import cleanup, digest, run_child
from run_rma_pipelines import read_status

STAGES = (
    ('dual_output', '旧B双输出测试（保存选定样本矩阵）', 'rma_dual_output.py', 'testing'),
    ('original_train', '第一组：原流程四适配器训练', 'train_rma_scratch_original.py', 'training'),
    ('original_test', '第一组：训练损失最佳权重测试', 'test_rma_scratch_original.py', 'testing'),
    ('new_train', '第二组：新流程四适配器训练', 'train_rma_scratch_new.py', 'training'),
    ('new_test', '第二组：源域验证最佳权重测试', 'test_rma_scratch_new.py', 'testing'),
)


def completed_session(receipt_file, entry_stem):
    """Use only this child's exact receipt; never search for the latest folder."""
    receipt = read_json(receipt_file)
    session = Path(receipt['session']).resolve()
    session.relative_to((ROOT/'results'/entry_stem).resolve())
    if receipt.get('phase') != 'completed' or read_json(session/'run_status.json').get('complete') is not True:
        raise RuntimeError(f'Child did not confirm completion: {session}')
    return session


def selected_checkpoint(session, flow):
    """Both test groups must use best.pth from their own completed training."""
    state = read_json(session/'run_status.json')
    manifest = read_json(session/'model_manifest.json')
    selected = Path(state['selected_checkpoint']).resolve()
    if (state.get('complete') is not True or state.get('flow') != flow or manifest.get('flow') != flow
            or manifest.get('format') != 'adaptclip-rma-scratch-v1'
            or selected.parent != session.resolve() or selected.name != 'best.pth' or not selected.is_file()):
        raise ValueError('Training did not provide its own completed matching best.pth.')
    return selected


def inspect_stop(receipt_file, phase):
    """Report last persisted epoch or dataset/shot without inventing progress."""
    receipt = read_status(receipt_file)
    result = dict(phase=phase, session=receipt.get('session'), receipt_phase=receipt.get('phase','not_reported'))
    if not result['session']:
        return result
    session = Path(result['session'])
    state = read_status(session/'run_status.json')
    result.update(child_status=state, log_directory=str(session))
    if phase == 'training':
        last_epoch = 0
        try:
            with (session/'training_history.csv').open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    last_epoch = max(last_epoch,int(row['epoch']))
        except (OSError,ValueError,csv.Error):
            pass
        result.update(last_recorded_epoch=last_epoch, total_epochs=receipt.get('epochs'))
    else:
        jobs = receipt.get('evaluation_jobs',[])
        done = sum(read_status(session/d/f'{s}shot_seed10'/'run_status.json').get('complete') is True for d,s in jobs)
        result.update(current=state.get('current'), completed_rounds=done, total_rounds=len(jobs))
    return result


def resume_stages(previous):
    """Reuse only a verified completed prefix; restart an unfinished stage fresh.

    This is stage-level continuation, not optimizer/epoch checkpoint resumption.
    Previous results and failure receipts are never changed.
    """
    previous = project_path(previous, ROOT)
    previous.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    completed = read_json(previous/'run_status.json')['completed_stages']
    if completed != [s[0] for s in STAGES[:len(completed)]] or len(completed) > len(STAGES):
        raise ValueError('Only a contiguous completed stage prefix can be reused.')
    receipts, training = {}, {}
    for key, _, filename, phase in STAGES[:len(completed)]:
        receipt_path = previous/f'{key}_progress.json'
        session = completed_session(receipt_path, Path(filename).stem)
        receipts[key] = read_json(receipt_path)
        if phase == 'training':
            flow = key.removesuffix('_train')
            selected_checkpoint(session,flow)
            training[flow] = session
        else:
            campaign = read_json(session/'campaign_config.json')
            datasets = ['Visa','BTAD','MPDD','SDD','DTD','part_glass']
            jobs = [[d,s] for d in datasets for s in (1,2,4)]
            if campaign['datasets'] != datasets or campaign['shots'] != [1,2,4] or campaign['seed'] != 10:
                raise ValueError('Completed test scope differs from this sequence.')
            if read_json(session/'run_status.json')['completed_jobs'] != jobs:
                raise ValueError('Completed test job list is incomplete.')
            for d,s in jobs:
                state = read_json(session/d/f'{s}shot_seed10'/'run_status.json')
                if state.get('complete') is not True or state.get('baseline_parity_passed') is not True:
                    raise ValueError('Completed test round failed parity/completion checks.')
                if key == 'dual_output' and not all(state.get(k) is True for k in ('old_B_parity_passed','dual_output_parity_passed')):
                    raise ValueError('Dual-output parity is incomplete.')
            if digest(campaign['checkpoint']) != campaign['checkpoint_sha256']:
                raise ValueError('Completed test checkpoint changed.')
            if key.endswith('_test'):
                flow = key.removesuffix('_test')
                if Path(campaign['checkpoint']).resolve() != selected_checkpoint(training[flow],flow):
                    raise ValueError('Completed test used a different training checkpoint.')
    return previous, list(completed), receipts, training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume_from', help='Previous sequence directory: reuse verified completed stages, restart the failed stage in a fresh directory.')
    args = parser.parse_args(argv)
    for _, _, filename, _ in STAGES:
        if not (ROOT/'models_test'/filename).is_file():
            raise FileNotFoundError(filename)
    previous, completed, reused_receipts, train_sessions = None, [], {}, {}
    if args.resume_from:
        previous, completed, reused_receipts, train_sessions = resume_stages(args.resume_from)
    output = new_directory(ROOT/'results'/Path(__file__).stem, 'dual_original_new_seed10')
    for key, value in reused_receipts.items():
        write_json(output/f'{key}_progress.json',value)
    write_json(output/'execution_plan.json', dict(stages=[dict(stage=k,label=l,script=f,phase=p) for k,l,f,p in STAGES],
        datasets=['Visa','BTAD','MPDD','SDD','DTD','part_glass'], shots=[1,2,4], seed=10,
        dual_output_save_maps=True, scratch_test_save_maps=False,
        resume_from=str(previous) if previous else None, reused_stages=list(completed),
        cache_policy='training epoch cleanup; isolated dataset/shot children; wait for full stage process exit'))
    reused_stages = set(completed)
    current, receipt, phase = None, None, None
    try:
        for index,(current,label,filename,phase) in enumerate(STAGES,1):
            if current in reused_stages:
                print(f'[{index}/{len(STAGES)}] 已核对并复用完成阶段：{label}',flush=True)
                continue
            receipt = output/f'{current}_progress.json'
            command = [sys.executable,'-u',str(ROOT/'models_test'/filename),'--_progress_file',str(receipt)]
            if current == 'dual_output':
                command.append('--save_maps')
            elif current.endswith('_test'):
                flow = current.removesuffix('_test')
                command.extend(['--checkpoint',str(selected_checkpoint(train_sessions[flow],flow))])
                if current == 'original_test' and previous is not None:
                    interrupted = read_status(previous/'original_test_progress.json')
                    if interrupted.get('session'):
                        command.extend(['--resume_test_run',str(interrupted['session'])])
            write_json(output/'run_status.json', dict(status='running',complete=False,current_stage=current,
                       current_label=label,completed_stages=completed,progress_file=str(receipt)))
            write_json(output/f'{current}_command.json',dict(command=command))
            print(f'[{index}/{len(STAGES)}] {label}',flush=True)
            try:
                run_child(command)
            finally:
                cleanup()
            session = completed_session(receipt,Path(filename).stem)
            if current.endswith('_train'):
                flow = current.removesuffix('_train')
                selected_checkpoint(session,flow)
                train_sessions[flow] = session
            completed.append(current)
        write_json(output/'run_status.json',dict(status='completed',complete=True,completed_stages=completed))
    except BaseException as error:
        try:
            progress = inspect_stop(receipt,phase)
        except Exception as inspect_error:
            progress = dict(progress_read_error=str(inspect_error))
        lines = [f'运行停止，后续任务未启动。当前阶段：{current}',f'已完成：{completed}',f'总记录：{output}']
        if progress.get('session'):
            lines.append(f'当前子任务目录：{progress["session"]}')
        if 'last_recorded_epoch' in progress:
            lines.append(f'最近完整记录的训练epoch：{progress["last_recorded_epoch"]}/{progress["total_epochs"]}')
        if 'completed_rounds' in progress:
            lines.append(f'已完成测试：{progress["completed_rounds"]}/{progress["total_rounds"]}；最后记录条件：{progress["current"]}')
        lines.append(f'错误/中断：{str(error) or "Ctrl+C"}')
        summary = '\n'.join(lines)
        print(summary,file=sys.stderr,flush=True)
        try:
            write_json(output/'run_status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                complete=False,current_stage=current,completed_stages=completed,error=str(error),stop_progress=progress))
            (output/'stop_summary.txt').write_text(summary+'\n',encoding='utf-8')
        except OSError as save_error:
            print(f'无法保存停止记录：{save_error}',file=sys.stderr,flush=True)
        raise
    print(f'全部五阶段完成。总记录：{output}',flush=True)


if __name__ == '__main__':
    main()
