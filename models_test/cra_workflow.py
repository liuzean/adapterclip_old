"""Shared two-stage launch/status/resume logic for CRA and ALIGN->CRA.

Every stage waits for its child process to exit. No torch/model imports here.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from cra_utils import checked_checkpoint, code_hashes, source_plan, verify_plan
from fusion_ablation_utils import new_directory, project_path, read_json
from rma_align_ablation_utils import DATASETS
from rma_control_io import write_json
from rma_utils import cleanup, run_child

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = {
    'cra_pipeline': (('train', 'train_cra.py'), ('test', 'test_cra.py')),
    'run_align_cra_sequence': (('align', 'rma_align_ablation.py'), ('cra', 'cra_pipeline.py')),
}


def safe_json(path):
    try: return read_json(path)
    except (OSError, ValueError): return {}


def describe(session):
    """Only saved stage, train epoch/batch or test dataset/shot; no liveness claim."""
    parts, visited = [], set()
    while str(session) not in visited:
        visited.add(str(session))
        state = safe_json(session/'run_status.json')
        if state.get('complete'): return ' → '.join(parts+['已完成'])
        training = safe_json(session/'training_progress.json')
        if training:
            return ' → '.join(parts+[f'训练：epoch={training.get("epoch")}，batch={training.get("batch")}'])
        if state.get('current'):
            dataset, _, shot = state['current'].partition('/')
            return ' → '.join(parts+[f'测试：数据集={dataset}，shot={shot}'])
        stage = state.get('current_stage')
        if not stage: return ' → '.join(parts+['尚无子任务进度'])
        parts.append('阶段='+stage)
        receipt = safe_json(session/f'{stage}_progress.json')
        if not receipt.get('session'): return ' → '.join(parts+['尚无子任务目录'])
        session = Path(receipt['session'])
    raise ValueError('Cyclic progress receipt.')


def require_complete(script, folder):
    """Check exact child receipts, not the newest directory or exit code alone."""
    folder = Path(folder).resolve(); stem = Path(script).stem
    folder.relative_to((ROOT/'results'/stem).resolve())
    status = read_json(folder/'run_status.json')
    if status.get('complete') is not True: raise ValueError('Incomplete child: '+str(folder))
    if stem == 'train_cra':
        checked_checkpoint(folder/'best.pth')
    elif stem in ('test_cra', 'rma_align_ablation'):
        plan = read_json(folder/'campaign_config.json')
        expected = {f'{d}/{s}' for d in DATASETS for s in (1, 2, 4)}
        if set(plan['jobs']) != expected or set(status['completed_jobs']) != expected or len(status['completed_jobs']) != 18:
            raise ValueError('Child must complete all18 dataset/shot conditions.')
        if stem == 'test_cra':
            from test_cra import verified_complete
            verify_plan(plan)
        else:
            from rma_align_ablation import verified_complete, verify_files
            verify_files(plan)
        for key, spec in plan['jobs'].items():
            child = Path(status['job_results'][key]); child.resolve().relative_to(folder)
            verified_complete(child, spec['queries'])
    elif stem == 'cra_pipeline':
        verify_plan(read_json(folder/'execution_plan.json'))
        if status['completed_stages'] != ['train', 'test']: raise ValueError('CRA pipeline stages incomplete.')
        for stage, child in SCHEDULES[stem]: require_complete(child, status['stage_results'][stage])
    else:
        raise ValueError('Unexpected workflow child: '+stem)


def initial_plan(stem):
    from rma_align_ablation import preflight
    checkpoint = ROOT/'results/train_rma_scratch_new/mvtec_seed10/best.pth'
    _, _, _, source = source_plan(checkpoint)
    targets = preflight(SimpleNamespace(checkpoint=checkpoint,
        reference_run=ROOT/'results/rma_score_coordination_inference/experiment_H_inference_seed10',
        datasets=list(DATASETS), shots=[1, 2, 4], num_workers=0, status=None, resume_from=None))
    return dict(experiment='K' if stem == 'cra_pipeline' else 'J_then_K',
        stages=SCHEDULES[stem], seed=10, datasets=DATASETS, shots=[1, 2, 4],
        code_sha256=code_hashes(), input_sha256=dict(source['input_sha256'], **targets['input_sha256']),
        note='ALIGN outcome does not alter CRA settings. Only CRA trains; no TGA/SPL/prompt changes.')


def pipeline_main(stem, argv=None):
    p = argparse.ArgumentParser(description='Run ordered experiment stages; stop on failure.', allow_abbrev=False)
    p.add_argument('--status'); p.add_argument('--resume_from'); p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    tokens = sys.argv[1:] if argv is None else argv
    if args.status and args.resume_from: p.error('Use status or resume separately.')
    if args.resume_from and any(t.startswith('--') and t.split('=', 1)[0] != '--resume_from' for t in tokens):
        p.error('Resume restores saved settings; use --resume_from alone.')
    result_root = ROOT/'results'/stem
    if args.status:
        session = project_path(args.status, ROOT); session.relative_to(result_root.resolve())
        print('只读取最后保存进度，不启动模型，也不据此判断进程是否仍在运行。')
        print(describe(session)); print('总记录：', session); return
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    if args.resume_from:
        session = project_path(args.resume_from, ROOT); session.relative_to(result_root.resolve())
        plan, status = read_json(session/'execution_plan.json'), read_json(session/'run_status.json')
        verify_plan(plan)
    else:
        print('启动前核对：H权重、源域episode及六集18条件历史输入。', flush=True)
        plan = initial_plan(stem)
        session = new_directory(result_root, 'K_seed10' if stem == 'cra_pipeline' else 'J_then_K_seed10')
        write_json(session/'execution_plan.json', plan)
        status = dict(complete=False, current_stage=None, completed_stages=[], stage_results={})
        write_json(session/'run_status.json', status)
    if receipt: write_json(receipt, dict(session=str(session), phase='running'))
    try:
        for index, (stage, script) in enumerate(SCHEDULES[stem], 1):
            status.update(complete=False, current_stage=stage)
            child_receipt = session/f'{stage}_progress.json'
            previous = status['stage_results'].get(stage) or safe_json(child_receipt).get('session')
            if previous and safe_json(Path(previous)/'run_status.json').get('complete') is True:
                require_complete(script, previous)
                status['stage_results'][stage] = previous
                if stage not in status['completed_stages']: status['completed_stages'].append(stage)
                write_json(session/'run_status.json', status)
                print('已核对并复用完成阶段：', stage, flush=True); continue
            verify_plan(plan)
            command = [sys.executable, '-u', str(ROOT/'models_test'/script)]
            if previous:
                Path(previous).resolve().relative_to((ROOT/'results'/Path(script).stem).resolve())
                command += ['--resume_from', previous]
            else:
                command += ['--_progress_file', str(child_receipt)]
                if script == 'test_cra.py':
                    selected = Path(status['stage_results']['train'])/'best.pth'
                    checked_checkpoint(selected); command += ['--checkpoint', str(selected)]
            if stage in status['completed_stages']:
                status['completed_stages'].remove(stage)
            status.pop('error', None); write_json(session/'run_status.json', status)
            write_json(session/f'{stage}_command.json', dict(command=command))
            print(f'[{index}/2] {stage}：{script}\n总记录：{session}', flush=True)
            try: run_child(command)
            finally: cleanup()
            child = safe_json(child_receipt).get('session') or previous
            if not child: raise ValueError('Child did not record its exact result directory.')
            require_complete(script, child)
            status['stage_results'][stage] = child; status['completed_stages'].append(stage)
            write_json(session/'run_status.json', status)
        verify_plan(plan)
        status.update(complete=True, current_stage=None); status.pop('error', None)
        write_json(session/'run_status.json', status)
        if receipt: write_json(receipt, dict(session=str(session), phase='completed'))
        print('所有阶段完成：', session, flush=True)
    except BaseException as error:
        status.update(complete=False, error=str(error) or 'KeyboardInterrupt')
        write_json(session/'run_status.json', status)
        text = '运行停止，后续阶段不启动。\n'+describe(session)+'\n总记录：'+str(session)
        (session/'stop_summary.txt').write_text(text+'\n', encoding='utf-8')
        print(text, file=sys.stderr, flush=True)
        raise
