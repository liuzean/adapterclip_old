r"""先准备外部评估，再串行完成实验M；失败停止并报告中断位置。

项目根目录运行：
    python .\models_test\run_paper_pqa_sequence.py
查看最后保存进度（不启动实验）：
    python .\models_test\run_paper_pqa_sequence.py --status .\results\run_paper_pqa_sequence\paper_M_seed10
继续同一会话：
    python .\models_test\run_paper_pqa_sequence.py --resume_from .\results\run_paper_pqa_sequence\paper_M_seed10

重复开始会创建编号目录；恢复时使用实际目录。M沿用原入口的完整epoch恢复、
测试逐dataset/shot恢复。突然关机无法即时打印，重启后用--status读取最后记录。
不自动执行官方SubspaceAD/DictAS复现，不修改原有训练、测试或实验代码。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json
from pqa_local_control_utils import (DATASETS, DUAL_RUN, REFERENCE_RUN, code_hashes,
                                     target_inputs, training_plan, verify_plan)
from rma_control_io import write_json
from rma_utils import cleanup, digest, run_child

RESULT_ROOT = ROOT/'results'/Path(__file__).stem
STAGES = ('prepare_paper_evaluation', 'pqa_local_control')
PREPARE_PREFIX = '外部评估准备完成；未启动外部复现：'


def implementation_hashes():
    return dict(code_hashes(), **{str(path):digest(path) for path in
        (Path(__file__).resolve(), ROOT/'models_test/prepare_paper_evaluation.py')})


def check_session(session):
    session = project_path(session, ROOT)
    session.relative_to(RESULT_ROOT.resolve())
    return session


def show_progress(session, state):
    print('总记录：', session, flush=True)
    print('当前阶段：', state.get('current'), '；阶段状态：', state.get('phase'), flush=True)
    print('已完成：', state.get('completed', []), flush=True)
    if state.get('error'): print('错误/中断：', state['error'], flush=True)
    child = state.get('control_session')
    if state.get('current') == STAGES[1] and child:
        child = Path(child).resolve()
        child.relative_to((ROOT/'results/pqa_local_control_pipeline').resolve())
        if (child/'run_status.json').is_file():
            from pqa_local_control_pipeline import show_progress as show_control_progress
            show_control_progress(child, read_json(child/'run_status.json'))
    elif state.get('current') == STAGES[0]:
        print('外部评估准备尚未完整完成；此阶段没有训练epoch或测试shot。', flush=True)
        if state.get('prepare_log'): print('准备阶段日志：', state['prepare_log'], flush=True)


def verified_preparation(state):
    directory = Path(state['prepare_output']).resolve()
    directory.relative_to((ROOT/'results/prepare_paper_evaluation').resolve())
    status = read_json(directory/'run_status.json')
    if status.get('complete') is not True or status.get('preparation_only') is not True or status.get('jobs') != 18:
        raise ValueError('External preparation is incomplete.')
    receipts = state.get('prepare_artifacts', {})
    if not {'run_status.json', 'protocol.json', 'jobs.csv', 'external_metrics_template.csv'} <= receipts.keys():
        raise ValueError('Missing external preparation artifact receipts.')
    for relative, sha in receipts.items():
        path = (directory/relative).resolve(); path.relative_to(directory)
        if digest(path) != sha: raise ValueError('External preparation output changed: '+relative)


def prepare(session, state):
    """Run the original entry unchanged; keep its console output and result path."""
    log = session/f'prepare_attempt_{state["prepare_attempt"]:02d}.log'
    state.update(prepare_log=str(log), phase='running')
    write_json(session/'run_status.json', state)
    command = [sys.executable, '-u', str(ROOT/'models_test/prepare_paper_evaluation.py')]
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    output = None
    with log.open('x', encoding='utf-8') as stream:
        with subprocess.Popen(command, cwd=str(ROOT), env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True); stream.write(line); stream.flush()
                    if line.startswith(PREPARE_PREFIX):
                        output = Path(line[len(PREPARE_PREFIX):].strip()).resolve()
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
            finally:
                child.stdout.close()
    if code: raise RuntimeError(f'External preparation failed with exit code {code}; experiment M not started.')
    if output is None: raise ValueError('Preparation exited without its completion/result-path receipt.')
    output.relative_to((ROOT/'results/prepare_paper_evaluation').resolve())
    state.update(prepare_output=str(output), prepare_artifacts={str(p.relative_to(output)):digest(p)
        for p in output.rglob('*') if p.is_file()})
    verified_preparation(state)


def create_control_session():
    """Use the existing M preflight/schema, so its session is known before launch."""
    _, _, _, plan = training_plan()
    for dataset in DATASETS:
        for shot in (1, 2, 4): target_inputs(REFERENCE_RUN, DUAL_RUN, dataset, shot)
    directory = new_directory(ROOT/'results/pqa_local_control_pipeline', 'M_seed10')
    write_json(directory/'campaign_config.json', plan)
    write_json(directory/'run_status.json', dict(complete=False, current=None, completed=[]))
    return directory


def verified_control(directory):
    from pqa_local_control_pipeline import STAGES as control_stages, verified_stage
    directory = Path(directory).resolve()
    directory.relative_to((ROOT/'results/pqa_local_control_pipeline').resolve())
    verify_plan(read_json(directory/'campaign_config.json'))
    state = read_json(directory/'run_status.json')
    if state.get('complete') is not True or state.get('completed') != list(control_stages):
        raise ValueError('Experiment M has not completed both training arms and full testing.')
    for stage in control_stages: verified_stage(directory, stage)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--status'); p.add_argument('--resume_from'); args = p.parse_args(argv)
    if args.status and args.resume_from: p.error('Use --status or --resume_from separately.')
    if args.status:
        session = check_session(args.status)
        print('只显示最后保存进度，不启动任务，也不据此判断进程是否仍在运行。')
        show_progress(session, read_json(session/'run_status.json')); return
    if args.resume_from:
        session = check_session(args.resume_from)
        plan, state = read_json(session/'campaign_config.json'), read_json(session/'run_status.json')
        if plan['code_sha256'] != implementation_hashes(): raise ValueError('Sequence implementation changed; cannot silently mix campaigns.')
        if state.get('completed') not in ([], [STAGES[0]], list(STAGES)):
            raise ValueError('Unexpected completed stage order.')
    else:
        session = new_directory(RESULT_ROOT, 'paper_M_seed10')
        plan = dict(stages=list(STAGES), code_sha256=implementation_hashes(), seed=10)
        write_json(session/'campaign_config.json', plan)
        state = dict(complete=False, current=None, phase='initializing', completed=[], prepare_attempt=0)
        write_json(session/'run_status.json', state)
    try:
        for stage in STAGES:
            state.update(complete=False, current=stage, phase='checking'); state.pop('error', None)
            write_json(session/'run_status.json', state)
            if stage in state['completed']:
                if stage == STAGES[0]: verified_preparation(state)
                else: verified_control(state['control_session'])
                print('核对并复用完成阶段：', stage, flush=True); continue
            if stage == STAGES[0]:
                state['prepare_attempt'] += 1
                print('[1/2] 外部评估准备', flush=True)
                prepare(session, state)
            else:
                print('[2/2] 实验M：无CRA训练 → 冻结CRA训练 → 四条件测试', flush=True)
                if not state.get('control_session'):
                    state.update(phase='preflight'); write_json(session/'run_status.json', state)
                    state['control_session'] = str(create_control_session())
                state.update(phase='running'); write_json(session/'run_status.json', state)
                run_child([sys.executable, '-u', str(ROOT/'models_test/pqa_local_control_pipeline.py'),
                           '--resume_from', state['control_session']])
                verified_control(state['control_session'])
            cleanup()
            state['completed'].append(stage); state['phase'] = 'stage_completed'
            write_json(session/'run_status.json', state)
        state.update(complete=True, current=None, phase='completed')
        write_json(session/'run_status.json', state)
        print('外部准备与实验M均已完成：', session, flush=True)
    except BaseException as error:
        state.update(complete=False, phase='stopped', error=str(error) or type(error).__name__)
        write_json(session/'run_status.json', state)
        print('运行停止，后续任务未启动。', flush=True)
        try: show_progress(session, state)
        except (OSError, ValueError, KeyError) as detail_error:
            print('未能读取更细进度，以上为总阶段记录：', detail_error, flush=True)
        raise


if __name__ == '__main__': main()
