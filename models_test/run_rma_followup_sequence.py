"""依次执行固定比例完整消融、MVTec局部诊断、β4/16/64目标消融。

沿用三项入口默认seed10/1,2,4-shot；不训练。等待每阶段退出后再启动下一项。
失败/Ctrl+C停止，记录当前阶段及子任务dataset/shot；--status只读，不续跑。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json, write_json
from rma_utils import cleanup, failure_status, run_child

STAGES = (
    ('fixed_fraction_grid', '固定比例完整目标消融', 'rma_fixed_fusion.py', ('--evaluate_all_fractions',), 'rma_fixed_fusion'),
    ('source_local_diagnostics', 'MVTec源域局部误差归因', 'rma_local_diagnostics.py', (), 'rma_local_diagnostics'),
    ('beta_target_ablation', 'β4/16/64完整目标消融', 'rma_reliability_beta_ablation.py', (), 'rma_reliability_beta_ablation'),
)
EXPECTED_JOBS = {(d, s) for d in ('Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass') for s in (1, 2, 4)}


def progress(receipt, result_folder):
    """Use exact child receipt, never guess latest output directory."""
    if not receipt.exists():
        return dict(note='子任务尚未写入目录回执，可能处于启动前检查。')
    report = read_json(receipt)
    session = project_path(report['session'], ROOT)
    session.relative_to((ROOT/'results'/result_folder).resolve())
    state_file = session/'run_status.json'
    return dict(session=str(session), child_status=read_json(state_file) if state_file.exists() else {},
                note=None if state_file.exists() else '子任务已创建目录，尚未写入运行状态。')


def require_complete(stage, report):
    state = report.get('child_status', {})
    if state.get('complete') is not True:
        raise ValueError('Child process exited without a completed run status.')
    if stage == 'source_local_diagnostics':
        if set(state.get('completed_shots', [])) != {1, 2, 4}:
            raise ValueError('Source diagnosis must complete all three shots.')
    else:
        if state.get('target_evaluation_complete') is not True or {tuple(r) for r in state.get('completed_jobs', [])} != EXPECTED_JOBS:
            raise ValueError('Target stage must finish all18 dataset/shot conditions.')
        if stage == 'beta_target_ablation' and set(state.get('betas', [])) != {4., 16., 64.}:
            raise ValueError('All three betas must finish.')


def describe(output, state, report):
    child = report.get('child_status', {})
    lines = [f'当前阶段：{state.get("current_label")}（{state.get("current_stage")}）',
        f'已完成阶段：{state.get("completed_stages", [])}', f'总记录：{output}']
    if report.get('session'):
        lines.append(f'当前子任务目录：{report["session"]}')
    if child.get('current'):
        lines.append(f'最近记录的数据集/shot：{child["current"]}')
    if child.get('current_shot') is not None:
        lines.append(f'最近记录的源域shot：{child["current_shot"]}')
    for key in ('completed_jobs', 'completed_shots'):
        if key in child:
            lines.append(f'子任务已完成：{key}={child[key]}')
    if report.get('note'):
        lines.append(report['note'])
    if child.get('error'):
        lines.append(f'子任务错误：{child["error"]}')
    if state.get('error'):
        lines.append(f'停止原因：{state["error"]}')
    return '\n'.join(lines)


def inspect(output):
    state = read_json(output/'run_status.json')
    if state.get('complete') is True:
        print(f'三项任务全部完成。总记录：{output}', flush=True); return
    stage = next(row for row in STAGES if row[0] == state['current_stage'])
    try:
        report = progress(output/f'{stage[0]}_progress.json', stage[4])
    except Exception as error:
        report = dict(note=f'读取子任务进度失败：{error}')
    print('以下仅为最后保存的进度，不启动或恢复任务，也不能据此判断进程是否仍在运行。', flush=True)
    print(describe(output, state, report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--status', help='Read saved status only; no execution/resume.')
    args = parser.parse_args(argv)
    if args.status:
        inspect(project_path(args.status, ROOT)); return
    for _, _, script, _, _ in STAGES:
        if not (ROOT/'models_test'/script).is_file():
            raise FileNotFoundError(script)
    output = new_directory(ROOT/'results'/Path(__file__).stem, 'three_tasks_seed10')
    write_json(output/'execution_plan.json', dict(stages=[dict(stage=k, label=l, script=s, arguments=a) for k, l, s, a, _ in STAGES],
        seed=10, shots=[1, 2, 4], no_training=True, automatic_resume=False))
    done = []; results = {}; current = None; label = None; receipt = None; folder = None
    try:
        for index, (current, label, script, options, folder) in enumerate(STAGES, 1):
            receipt = output/f'{current}_progress.json'
            command = [sys.executable, '-u', str(ROOT/'models_test'/script), *options, '--_progress_file', str(receipt)]
            write_json(output/'run_status.json', dict(status='running', complete=False, current_stage=current,
                current_label=label, completed_stages=done, stage_results=results))
            write_json(output/f'{current}_command.json', dict(command=command))
            print(f'[{index}/3] {label}\n总记录：{output}', flush=True)
            try:
                run_child(command)
            finally:
                cleanup()
            report = progress(receipt, folder); require_complete(current, report)
            done.append(current); results[current] = report['session']
            write_json(output/'run_status.json', dict(status='stage_completed', complete=False, current_stage=current,
                current_label=label, completed_stages=done, stage_results=results))
        write_json(output/'run_status.json', dict(status='completed', complete=True, completed_stages=done, stage_results=results))
    except BaseException as error:
        try:
            report = progress(receipt, folder) if receipt is not None else {}
        except Exception as inspection_error:
            report = dict(note=f'子任务进度读取失败：{inspection_error}')
        write_json(output/'run_status.json', dict(status='running', complete=False, current_stage=current,
            current_label=label, completed_stages=done, stage_results=results, stop_progress=report))
        failure_status(output, error)
        text = '运行停止，后续阶段未启动。\n'+describe(output, read_json(output/'run_status.json'), report)
        (output/'stop_summary.txt').write_text(text+'\n', encoding='utf-8')
        print(text, file=sys.stderr, flush=True); raise
    print(f'三项任务全部完成。总记录：{output}', flush=True)


if __name__ == '__main__':
    main()
