"""依次运行新D关闭RMA测试、三适配器训练、该次best权重测试。

各测试默认六数据集1/2/4-shot、seed10；每阶段/条件独立子进程，等退出后继续。
失败或Ctrl+C停止后续任务，输出当前阶段及最近保存的epoch或数据集/shot。
父调度器不导入torch，不自动续跑，不覆盖已有实验结果。
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json, write_json
from rma_utils import cleanup, digest, run_child
from run_rma_experiment_sequence import completed_session, inspect_stop
from run_rma_pipelines import read_status

STAGES=(
    ('without_rma','新D第14轮同权重关闭RMA测试','new_d_without_rma.py','testing'),
    ('three_train','新流程三适配器训练','train_new_d_three.py','training'),
    ('three_test','新流程三适配器最佳权重测试','test_new_d_three.py','testing'),
)


def selected_checkpoint(session):
    """Use only the selected best.pth from this invocation's completed training."""
    session=Path(session).resolve()
    state=read_json(session/'run_status.json')
    manifest=read_json(session/'model_manifest.json')
    checkpoint=Path(state['selected_checkpoint']).resolve()
    if (state.get('complete') is not True or state.get('flow')!='new'
            or state.get('three_adapter_gradient_checks_passed') is not True
            or manifest.get('format')!='adaptclip-three-scratch-v1'
            or manifest.get('initialization')!='CLIP pretrained; three adapters fresh; no RMA'
            or checkpoint.parent!=session or checkpoint.name!='best.pth' or not checkpoint.is_file()):
        raise ValueError('Expected this completed three-adapter training session and its selected best.pth.')
    return checkpoint


def progress_text(stage,label,completed,output,progress,error=None):
    lines=[f'当前阶段：{label}（{stage}）',f'已完成阶段：{", ".join(completed) or "无"}',f'总记录：{output}']
    if progress.get('session'):
        lines.append(f'当前子任务目录：{progress["session"]}')
    else:
        lines.append('子任务尚未写入结果目录回执，可能处于启动前检查。')
    if 'last_recorded_epoch' in progress:
        lines.append(f'最近完整记录的训练epoch：{progress["last_recorded_epoch"]}/{progress.get("total_epochs") or "未知"}')
    if 'completed_rounds' in progress:
        lines.append(f'已完成测试条件：{progress["completed_rounds"]}/{progress["total_rounds"]}')
        if progress.get('current'):
            lines.append(f'最后记录的数据集/shot：{progress["current"]}')
    if progress.get('progress_read_error'):
        lines.append('进度文件读取失败：'+progress['progress_read_error'])
    if error is not None:
        lines.append(f'错误/中断：{str(error) or "Ctrl+C"}')
    return '\n'.join(lines)


def inspect_sequence(output):
    """Read persisted progress after a shutdown; do not infer whether it is running."""
    state=read_status(output/'run_status.json')
    if not state:
        raise ValueError('No readable sequence run_status.json found.')
    if state.get('complete') is True:
        print(f'三个阶段均已完成。总记录：{output}',flush=True)
        return
    stage=state.get('current_stage')
    matching=next((row for row in STAGES if row[0]==stage),None)
    if matching is None:
        raise ValueError('Unknown recorded sequence stage.')
    _,label,_,phase=matching
    progress=inspect_stop(output/f'{stage}_progress.json',phase)
    print('以下为最后保存的进度，不能据此判断进程目前是否仍在运行。',flush=True)
    print(progress_text(stage,label,state.get('completed_stages',[]),output,progress),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--status',help='Read saved sequence progress only; does not start or resume any task.')
    args=parser.parse_args(argv)
    if args.status:
        inspect_sequence(project_path(args.status,ROOT))
        return
    for _,_,filename,_ in STAGES:
        if not (ROOT/'models_test'/filename).is_file():
            raise FileNotFoundError(filename)
    output=new_directory(ROOT/'results'/Path(__file__).stem,'new_D_controls_seed10')
    write_json(output/'execution_plan.json',dict(stages=[dict(stage=k,label=l,script=f,phase=p) for k,l,f,p in STAGES],
        datasets=['Visa','BTAD','MPDD','SDD','DTD','part_glass'],shots=[1,2,4],seed=10,
        checkpoint_policy='Use exact selected best.pth reported by this three-adapter training; never guess latest folder.',
        cache_policy='Independent stage/condition processes; wait for exit; training cleanup each epoch.'))
    completed=[]
    training=None
    current,label,receipt,phase=None,None,None,None
    try:
        for index,(current,label,filename,phase) in enumerate(STAGES,1):
            receipt=output/f'{current}_progress.json'
            command=[sys.executable,'-u',str(ROOT/'models_test'/filename),'--_progress_file',str(receipt)]
            if current=='three_test':
                checkpoint=selected_checkpoint(training)
                command.extend(['--checkpoint',str(checkpoint)])
                write_json(output/'selected_checkpoint.json',dict(training_session=str(training),checkpoint=str(checkpoint),sha256=digest(checkpoint)))
            write_json(output/'run_status.json',dict(status='running',complete=False,current_stage=current,
                current_label=label,completed_stages=completed,progress_file=str(receipt)))
            write_json(output/f'{current}_command.json',dict(command=command))
            print(f'[{index}/{len(STAGES)}] {label}',flush=True)
            print(f'总记录：{output}',flush=True)
            try:
                run_child(command)
            finally:
                cleanup()
            session=completed_session(receipt,Path(filename).stem)
            if phase=='training':
                selected_checkpoint(session)
                training=session
            completed.append(current)
            write_json(output/'run_status.json',dict(status='running',complete=False,current_stage=current,
                current_label=label,completed_stages=completed,progress_file=str(receipt)))
        write_json(output/'run_status.json',dict(status='completed',complete=True,completed_stages=completed))
    except BaseException as error:
        try:
            progress=inspect_stop(receipt,phase)
        except Exception as inspection_error:
            progress=dict(progress_read_error=str(inspection_error))
        summary='运行停止，后续阶段未启动。\n'+progress_text(current,label,completed,output,progress,error)
        print(summary,file=sys.stderr,flush=True)
        try:
            write_json(output/'run_status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                complete=False,current_stage=current,current_label=label,completed_stages=completed,
                error=str(error),stop_progress=progress))
            (output/'stop_summary.txt').write_text(summary+'\n',encoding='utf-8')
        except OSError as save_error:
            print(f'无法保存停止记录：{save_error}',file=sys.stderr,flush=True)
        raise
    print(f'三个阶段全部完成。总记录：{output}',flush=True)


if __name__=='__main__':
    main()
