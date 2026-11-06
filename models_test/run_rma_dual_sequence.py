"""依次验证固定12.5%双输出和β4双输出；每轮进程退出清理后继续。

默认各六数据集1/2/4-shot、seed10；失败停止并报告阶段/dataset/shot。
--status只读进度，不启动或恢复实验。父进程不导入torch，不训练。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json, write_json
from rma_utils import cleanup, failure_status, run_child
from run_rma_followup_sequence import describe, progress
from run_rma_followup_sequence import require_complete as require_target_complete

STAGES = (
    ('fixed125', '固定12.5%双输出验证', 'rma_fixed_dual_output.py'),
    ('beta4', 'β4双输出验证', 'rma_beta4_dual_output.py'),
)


def require_complete(mode, report):
    """Require all18 rounds and dual/offline checks before the next experiment."""
    require_target_complete(mode, report)
    state = report['child_status']
    if state.get('mode') != mode or not all(state.get(k) is True for k in (
            'dual_output_parity_passed', 'offline_combination_parity_passed')):
        raise ValueError('Wrong dual mode or failed/incomplete dual/offline parity checks.')


def inspect(output):
    state = read_json(output/'run_status.json')
    if state.get('complete') is True:
        print(f'两项双输出验证全部完成。总记录：{output}', flush=True); return
    mode, _, script = next(row for row in STAGES if row[0] == state['current_stage'])
    try:
        report = progress(output/f'{mode}_progress.json', Path(script).stem)
    except Exception as error:
        report = dict(note=f'读取子任务进度失败：{error}')
    print('以下仅为最后保存的进度，不启动/续跑任务，也不能据此判断进程是否仍在运行。', flush=True)
    print(describe(output, state, report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--status', help='Read saved progress only; no execution or resume.')
    args = parser.parse_args(argv)
    if args.status:
        inspect(project_path(args.status, ROOT)); return
    for _, _, script in STAGES:
        if not (ROOT/'models_test'/script).is_file():
            raise FileNotFoundError(script)
    output = new_directory(ROOT/'results'/Path(__file__).stem, 'two_duals_seed10')
    write_json(output/'execution_plan.json', dict(stages=[dict(mode=m, label=l, script=s) for m, l, s in STAGES],
        seed=10, shots=[1, 2, 4], no_training=True, automatic_resume=False,
        cleanup='Each dataset/shot is an isolated child; wait for exit. Each stage exits before next stage.'))
    done = []; results = {}; current = label = script = receipt = None
    try:
        for index, (current, label, script) in enumerate(STAGES, 1):
            receipt = output/f'{current}_progress.json'
            command = [sys.executable, '-u', str(ROOT/'models_test'/script), '--_progress_file', str(receipt)]
            write_json(output/'run_status.json', dict(status='running', complete=False, current_stage=current,
                current_label=label, completed_stages=done, stage_results=results))
            write_json(output/f'{current}_command.json', dict(command=command))
            print(f'[{index}/2] {label}\n总记录：{output}', flush=True)
            try:
                run_child(command)
            finally:
                cleanup()
            report = progress(receipt, Path(script).stem); require_complete(current, report)
            done.append(current); results[current] = report['session']
            write_json(output/'run_status.json', dict(status='stage_completed', complete=False, current_stage=current,
                current_label=label, completed_stages=done, stage_results=results))
        write_json(output/'run_status.json', dict(status='completed', complete=True, completed_stages=done, stage_results=results))
    except BaseException as error:
        try:
            report = progress(receipt, Path(script).stem) if receipt is not None else {}
        except Exception as inspection_error:
            report = dict(note=f'子任务进度读取失败：{inspection_error}')
        write_json(output/'run_status.json', dict(status='running', complete=False, current_stage=current,
            current_label=label, completed_stages=done, stage_results=results, stop_progress=report))
        failure_status(output, error)
        text = '运行停止，后续阶段未启动。\n'+describe(output, read_json(output/'run_status.json'), report)
        (output/'stop_summary.txt').write_text(text+'\n', encoding='utf-8')
        print(text, file=sys.stderr, flush=True); raise
    print(f'两项双输出验证全部完成。总记录：{output}', flush=True)


if __name__ == '__main__':
    main()
