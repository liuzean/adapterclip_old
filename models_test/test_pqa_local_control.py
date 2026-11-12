r"""实验M测试：固定K图像分数，比较四个定位条件；六数据集1/2/4-shot。

python .\models_test\test_pqa_local_control.py --plain_run <目录> --cra_run <目录>
每dataset/shot独立子进程；进程退出释放本轮模型和缓存，再进行下一轮。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cra_dual_output_utils import CRA_SHA
from cra_utils import attach
from fusion_ablation_utils import new_directory, project_path, read_csv, read_json, write_csv
from h_inference import WEIGHT_SHA, load_model
from pqa_local_control_utils import (CONDITIONS, CRA_WEIGHT, DUAL_RUN, FORMAT, H_WEIGHT, REFERENCE_RUN,
    SETTINGS, checked_training, compare, summarize, target_inputs, target_plan, verify_plan)
from rma_control_io import write_json, write_progress
from rma_utils import check_frozen, cleanup, digest, freeze_base, run_child

RESULT_ROOT = ROOT/'results'/Path(__file__).stem


def verified_complete(folder, count):
    status = read_json(Path(folder)/'run_status.json')
    flags = ('complete', 'history_parity_passed', 'fixed_K_image_passed', 'frozen_integrity_passed')
    if not all(status.get(k) is True for k in flags) or status.get('processed_queries') != count:
        raise ValueError('Local-control test round incomplete or checks failed.')
    required = {'experiment_config.json','references.csv','sample_manifest.csv','metrics_combined.csv',
                'image_scores.csv','history_checks.csv','deltas.csv'}
    artifacts = status.get('artifact_sha256', {})
    if not required <= artifacts.keys(): raise ValueError('Missing test artifact receipts.')
    for name, sha in artifacts.items():
        if Path(name).name != name or digest(Path(folder)/name) != sha:
            raise ValueError('Saved test artifact changed: '+name)


def load_head(state, spec, expected_initial):
    import torch
    from pqa_local_control_model import make_classifier
    payload = torch.load(spec['path'], map_location='cpu', weights_only=True)
    if (payload.get('format') != FORMAT or payload.get('arm') != spec['arm'] or payload.get('epoch') != spec['epoch']
            or payload.get('campaign_sha256') != spec['campaign_sha256'] or payload.get('settings') != SETTINGS
            or payload.get('initial_classifier_sha256') != expected_initial
            or payload.get('h_checkpoint_sha256') != WEIGHT_SHA or payload.get('cra_checkpoint_sha256') != CRA_SHA):
        raise ValueError('Wrong local head payload, initialization, arm or selected epoch.')
    from train_pqa_local_control import tensor_digest
    head = make_classifier(state.adapters[2])
    if tensor_digest(head) != expected_initial: raise ValueError('Actual original classifier initialization differs.')
    head.load_state_dict(payload['classifier'], strict=True)
    return head.eval().requires_grad_(False)


def worker(args, session, plan):
    from pqa_local_control_model import FrozenPaths, evaluate
    key = f'{args._dataset}/{args._shot}'
    if key not in plan['jobs']: raise ValueError('Unscheduled test condition.')
    output = project_path(args._output, ROOT)
    if output.parent != session/args._dataset: raise ValueError('Test output escaped its dataset directory.')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, dataset=args._dataset, shot=args._shot, processed_queries=0)
    write_json(output/'run_status.json', status)
    state = engine = hook = None
    try:
        verify_plan(plan)
        inputs, _, historical, old_scores, original = target_inputs(
            plan['reference_run'], plan['dual_run'], args._dataset, args._shot)
        write_json(output/'experiment_config.json', dict(inputs['config'], experiment='M', settings=SETTINGS,
            weights=plan['weights'], conditions=CONDITIONS, fixed_image_donor='K_dual original local head',
            historical_pixel_arrays_compared=False))
        for name in ('references.csv', 'sample_manifest.csv'):
            rows = read_csv(inputs['history']/name); write_csv(output/name, list(rows[0]), rows)
        state = load_model(H_WEIGHT, inputs, output, 0)
        cra, hook = attach(state, CRA_WEIGHT)
        cra_snapshot = freeze_base([None, cra])
        engine = FrozenPaths(state, hook)
        statuses = {arm:read_json(Path(spec['path']).parent/'run_status.json') for arm,spec in plan['weights'].items()}
        initial = statuses['plain']['initial_classifier_sha256']
        if statuses['cra']['initial_classifier_sha256'] != initial:
            raise ValueError('Two adapted classifiers did not start from identical original weights.')
        heads = {arm:load_head(state, spec, initial) for arm,spec in plan['weights'].items()}
        def progress(batch, count):
            status.update(batch=batch, processed_queries=count); write_progress(output/'evaluation_progress.json', status)
        table = evaluate(state, engine, heads, state.loader, output, inputs['manifest'],
                         history_scores=old_scores, progress=progress)
        checks, deltas = compare(table, historical, original, inputs['classes'])
        write_csv(output/'history_checks.csv', list(checks[0]), checks)
        write_csv(output/'deltas.csv', list(deltas[0]), deltas)
        if not all(r['Passed'] for r in checks): raise ValueError('Frozen H/K donor metrics differ from history.')
        for snapshot in (*state.snapshots, cra_snapshot): check_frozen(snapshot)
        verify_plan(plan)
        names = ('experiment_config.json', 'references.csv', 'sample_manifest.csv', 'metrics_combined.csv',
                 'image_scores.csv', 'history_checks.csv', 'deltas.csv')
        status.update(complete=True, processed_queries=len(inputs['manifest']), history_parity_passed=True,
            fixed_K_image_passed=True, frozen_integrity_passed=True,
            artifact_sha256={n:digest(output/n) for n in names})
        write_json(output/'run_status.json', status)
    except BaseException as error:
        status.update(complete=False, error=str(error) or 'KeyboardInterrupt')
        write_json(output/'run_status.json', status); raise
    finally:
        if engine is not None: engine.close()
        if hook is not None: hook.close()
        state = engine = None; cleanup()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--plain_run', default='results/train_pqa_local_control/plain_seed10')
    p.add_argument('--cra_run', default='results/train_pqa_local_control/cra_seed10')
    p.add_argument('--reference_run', default=str(REFERENCE_RUN)); p.add_argument('--dual_run', default=str(DUAL_RUN))
    p.add_argument('--status'); p.add_argument('--resume_from')
    for n in ('session', 'output', 'dataset', 'shot', 'progress_file'):
        p.add_argument('--_'+n, type=int if n == 'shot' else str, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.status and a.resume_from: p.error('Use --status or --resume_from separately.')
    hidden = [getattr(a, '_'+n) is not None for n in ('session', 'output', 'dataset', 'shot')]
    if any(hidden) and (not all(hidden) or a.status or a.resume_from): p.error('Incomplete internal worker arguments.')
    tokens = sys.argv[1:] if argv is None else argv
    if a.resume_from and any(t.startswith('--') and t.split('=',1)[0] not in ('--resume_from','--_progress_file') for t in tokens):
        p.error('Resume restores saved settings; use --resume_from alone.')
    for n in ('plain_run', 'cra_run', 'reference_run', 'dual_run'): setattr(a,n,project_path(getattr(a,n),ROOT))
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        session = project_path(args.status, ROOT); session.relative_to(RESULT_ROOT.resolve())
        status = read_json(session/'run_status.json')
        print('只读最后保存进度，不启动模型；当前数据集/shot：', status.get('current'))
        print('已完成：', status.get('completed_jobs', [])); return
    if args._session:
        session = project_path(args._session, ROOT); session.relative_to(RESULT_ROOT.resolve())
        worker(args, session, read_json(session/'campaign_config.json')); return
    if args.resume_from:
        session = project_path(args.resume_from, ROOT); session.relative_to(RESULT_ROOT.resolve())
        plan, status = read_json(session/'campaign_config.json'), read_json(session/'run_status.json')
        verify_plan(plan)
        if set(status['job_results'])-set(plan['jobs']): raise ValueError('Resume contains unscheduled jobs.')
    else:
        plan = target_plan(args.plain_run, args.cra_run, args.reference_run, args.dual_run)
        session = new_directory(RESULT_ROOT, 'M_seed10')
        write_json(session/'campaign_config.json', plan)
        status = dict(complete=False, current=None, completed_jobs=[], job_results={})
        write_json(session/'run_status.json', status)
    if args._progress_file: write_json(project_path(args._progress_file,ROOT), dict(session=str(session), phase='testing'))
    try:
        for key, spec in plan['jobs'].items():
            dataset, shot = key.split('/')
            status.update(complete=False, current=key)
            previous = status['job_results'].get(key)
            if previous:
                old = Path(previous).resolve(); old.relative_to(session)
                if (old/'run_status.json').is_file() and read_json(old/'run_status.json').get('complete'):
                    verified_complete(old, spec['queries'])
                    if key not in status['completed_jobs']: status['completed_jobs'].append(key)
                    continue
            parent = session/dataset; parent.mkdir(parents=True, exist_ok=True)
            output, number = parent/f'{shot}shot_seed10', 2
            while output.exists():
                output = parent/f'{shot}shot_seed10_run{number:02d}'; number += 1
            status['job_results'][key] = str(output); status['current_output'] = str(output); status.pop('error',None)
            if key in status['completed_jobs']: status['completed_jobs'].remove(key)
            write_json(session/'run_status.json', status)
            print(f'M测试：{dataset}，{shot}-shot', flush=True)
            run_child([sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),
                       '--_output',str(output),'--_dataset',dataset,'--_shot',shot])
            verified_complete(output,spec['queries']); cleanup()
            status['completed_jobs'].append(key); write_json(session/'run_status.json', status)
        summarize(session,status); verify_plan(plan)
        status.update(complete=True,current=None); status.pop('error',None)
        write_json(session/'run_status.json', status)
        print('M六数据集测试完成：',session,flush=True)
    except BaseException as error:
        status.update(complete=False,error=str(error) or 'KeyboardInterrupt')
        write_json(session/'run_status.json', status)
        print(f'测试停止，后续不启动：数据集/shot={status.get("current")}\n{session}',flush=True); raise


if __name__ == '__main__': main()
