"""Run the complete B pipeline, then the complete C pipeline, sequentially.

From project root: python models_test/run_rma_pipelines.py
Each pipeline keeps its existing six-dataset, 1/2/4-shot, seed=10 defaults.
Only a successful child-process exit allows the next pipeline to start.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, read_json, write_json
from rma_utils import run_child


def read_status(path):
    """Missing/incomplete status files mean unknown progress, never success."""
    if path is None:
        return {}
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def inspect_progress(stage, receipt_path):
    """Inspect only this child's explicitly reported session, not latest folders."""
    receipt = read_status(receipt_path)
    result = {'stage': stage, 'phase': receipt.get('phase', 'not_reported')}
    for name in ('session', 'train_output', 'test_output'):
        if receipt.get(name):
            result[name] = receipt[name]
    result['total_epochs'] = receipt.get('epochs')
    train_dir = Path(result['train_output']) if 'train_output' in result else None
    test_dir = Path(result['test_output']) if 'test_output' in result else None
    session = Path(result['session']) if 'session' in result else None
    train_status = read_status(train_dir/'run_status.json' if train_dir else None)
    test_status = read_status(test_dir/'run_status.json' if test_dir else None)
    pipeline_status = read_status(session/'run_status.json' if session else None)
    # Training history is written only after that epoch's validation finishes.
    last_epoch = 0
    if train_dir:
        try:
            with (train_dir/'training_history.csv').open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    try:
                        last_epoch = max(last_epoch, int(row.get('epoch', '')))
                    except (ValueError, TypeError):
                        continue
        except (OSError, UnicodeError, csv.Error):
            pass
    result.update(last_validated_epoch=last_epoch, training_complete=train_status.get('complete') is True,
                  testing_complete=test_status.get('complete') is True)
    if train_dir:
        result['train_log'] = str(train_dir/'train.log')
    jobs = receipt.get('evaluation_jobs', [])
    worker_statuses = {}
    if test_dir:
        for dataset, shot in jobs:
            worker_statuses[(dataset, shot)] = read_status(test_dir/dataset/f'{shot}shot_seed10'/'run_status.json')
    result['completed_test_rounds'] = sum(s.get('complete') is True for s in worker_statuses.values())
    result['total_test_rounds'] = len(jobs)
    current = test_status.get('current')
    if not isinstance(current, list) or len(current) != 2 or tuple(current) not in worker_statuses:
        current = next(([d, s] for (d, s), state in worker_statuses.items()
                        if state and state.get('complete') is not True), None)
    if current is None and result['phase'] == 'testing':
        current = next(([d, s] for (d, s), state in worker_statuses.items()
                        if state.get('complete') is not True), None)
    worker = {}
    if current and test_dir:
        dataset, shot = current
        worker = worker_statuses[(dataset, shot)]
        result.update(dataset=dataset, shot=shot, round_status=worker.get('status', 'not_started_or_not_recorded'),
                      round_complete=worker.get('complete') is True,
                      round_output=str(test_dir/dataset/f'{shot}shot_seed10'))
    leaf = worker if result['phase'] == 'testing' else train_status
    result['recorded_error'] = leaf.get('error') or pipeline_status.get('error')
    return result


def stop_summary(progress, completed, output, error):
    """Human-readable last confirmed progress; never claims an unfinished epoch."""
    phases = {'preflight': '启动前检查', 'initializing': '创建运行目录/初始化',
              'training': '训练', 'testing': '测试', 'completed': '流程收尾',
              'not_reported': '子流程启动/尚未写入进度'}
    phase = progress['phase']
    label = phases.get(phase, phase)
    if phase == 'training' and progress.get('training_complete'):
        label = '训练已完成，测试前检查/衔接'
    elif phase == 'testing' and progress.get('testing_complete'):
        label = '测试已完成，流程收尾'
    lines = ['运行已中断，后续任务已停止。' if isinstance(error, KeyboardInterrupt)
             else '运行失败，后续任务已停止。',
             f'当前方案：{progress["stage"]}', f'已完成方案：{", ".join(completed) or "无"}',
             f'停止阶段：{label}']
    if phase == 'training':
        total = progress.get('total_epochs') or '未知'
        lines.append(f'最近记录的完整验证轮次：{progress["last_validated_epoch"]}/{total}（未完成轮次不计入）')
    if phase == 'testing':
        lines.append(f'已完成测试：{progress["completed_test_rounds"]}/{progress["total_test_rounds"]} 轮')
        if 'dataset' in progress:
            qualifier = ('已完成；可能停在调度/汇总阶段' if progress['round_complete'] else
                         '本轮未完成' if progress['round_status'] != 'not_started_or_not_recorded' else '尚无启动记录')
            lines.append(f'最后记录/下一未完成条件：{progress["dataset"]}，{progress["shot"]}-shot，seed=10（{qualifier}）')
            lines.append(f'本轮结果及日志目录：{progress["round_output"]}')
    if phase == 'training' and progress.get('train_log'):
        lines.append(f'训练日志：{progress["train_log"]}')
    if progress.get('session'):
        lines.append(f'当前子流程记录：{progress["session"]}')
    lines.append(f'总调度记录：{output}')
    reason = progress.get('recorded_error') or str(error) or '用户中断（Ctrl+C）'
    lines.append(f'错误/中断信息：{reason}')
    return '\n'.join(lines)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    stages = [('B', ROOT/'models_test/rma_pipeline.py'),
              ('C', ROOT/'models_test/rma_joint_pipeline.py')]
    for _, script in stages:
        if not script.is_file():
            raise FileNotFoundError(script)
    output = new_directory(ROOT/'results'/Path(__file__).stem, 'B_then_C_seed10')
    plan = [{'stage': name, 'progress_file': str(output/f'{name}_progress.json'),
             'command': [sys.executable, '-u', str(script), '--_progress_file', str(output/f'{name}_progress.json')]}
            for name, script in stages]
    write_json(output/'execution_plan.json', {
        'stages': plan, 'order': 'B train + test -> C train + test',
        'datasets': ['Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass'],
        'shots': [1, 2, 4], 'seed': 10,
        'outputs': 'Each pipeline retains its own existing results directories.',
    })
    completed, current, receipt = [], None, None
    try:
        for stage in plan:
            current = stage['stage']
            receipt = Path(stage['progress_file'])
            write_json(output/'run_status.json', {
                'status': 'running', 'complete': False,
                'current_stage': current, 'completed_stages': completed,
            })
            print(f'[{len(completed)+1}/2] Starting {current}: {stage["command"][2]}', flush=True)
            run_child(stage['command'])  # Wait for complete exit before advancing.
            completed.append(current)
        write_json(output/'run_status.json', {
            'status': 'completed', 'complete': True, 'completed_stages': completed,
        })
    except BaseException as error:
        try:
            progress = inspect_progress(current, receipt)
        except Exception as inspect_error:
            # Progress reporting must never mask the original child failure.
            progress = {'stage': current, 'phase': 'not_reported', 'progress_read_error': str(inspect_error)}
        summary = stop_summary(progress, completed, output, error)
        print('\n' + summary, file=sys.stderr, flush=True)
        try:
            write_json(output/'run_status.json', {
                'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                'complete': False, 'current_stage': current,
                'completed_stages': completed, 'error': str(error), 'stop_progress': progress,
            })
            write_json(output/'stop_progress.json', progress)
            (output/'stop_summary.txt').write_text(summary + '\n', encoding='utf-8')
        except OSError as save_error:
            print(f'无法写入停止记录：{save_error}', file=sys.stderr, flush=True)
        raise
    print(f'B and C pipelines completed. Execution record: {output}', flush=True)


if __name__ == '__main__':
    main()
