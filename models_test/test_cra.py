r"""实验K测试：H第14轮+仅CRA源域最优权重，六数据集各1/2/4-shot。

python .\models_test\test_cra.py --checkpoint .\results\train_cra\mvtec_seed10\best.pth
CRA关=原H；CRA开=同H公式的独立候选。保留ALIGN，不采用实验J的关闭结果。
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cra_utils import attach, checked_checkpoint, code_hashes, compare, evaluate, verify_plan
from fusion_ablation_utils import new_directory, project_path, read_csv, read_json, write_csv
from h_inference import load_model, recorded_inputs
from rma_align_ablation_utils import DATASETS, retention, verified_history
from rma_control_io import write_json, write_progress
from rma_utils import check_frozen, cleanup, digest, freeze_base, run_child

RESULT_ROOT = ROOT/'results'/Path(__file__).stem


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', default='results/train_cra/mvtec_seed10/best.pth')
    p.add_argument('--reference_run', default='results/rma_score_coordination_inference/experiment_H_inference_seed10')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--status'); p.add_argument('--resume_from')
    for name in ('session', 'output', 'dataset', 'shot', 'progress_file'):
        p.add_argument('--_'+name, help=argparse.SUPPRESS, type=int if name == 'shot' else str)
    args = p.parse_args(argv)
    if args.datasets != sorted(set(args.datasets), key=DATASETS.index) or args.shots != sorted(set(args.shots)) or args.num_workers < 0:
        p.error('Ordered unique datasets/shots and nonnegative workers required.')
    hidden = [getattr(args, '_'+n) is not None for n in ('session', 'output', 'dataset', 'shot')]
    if any(hidden) and not all(hidden): p.error('Incomplete internal worker fields.')
    tokens = sys.argv[1:] if argv is None else argv
    if args.status and args.resume_from: p.error('Use status or resume separately.')
    if args.resume_from and any(t.startswith('--') and t.split('=', 1)[0] != '--resume_from' for t in tokens):
        p.error('Resume restores saved settings; use --resume_from alone.')
    for name in ('checkpoint', 'reference_run'): setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def preflight(args):
    from rma_align_ablation import preflight as h_preflight
    training, selected, weight_hash = checked_checkpoint(args.checkpoint)
    hargs = SimpleNamespace(checkpoint=Path(training['h_checkpoint']), reference_run=args.reference_run,
        datasets=args.datasets, shots=args.shots, num_workers=args.num_workers, status=None, resume_from=None)
    plan = h_preflight(hargs)
    plan.update(experiment='K', code_sha256=code_hashes(), h_checkpoint=training['h_checkpoint'],
        cra_checkpoint=str(args.checkpoint), cra_checkpoint_sha256=weight_hash, cra_epoch=selected['best_epoch'],
        options=dict(checkpoint=str(args.checkpoint), reference_run=str(args.reference_run),
                     datasets=args.datasets, shots=args.shots, num_workers=args.num_workers),
        image_score_source='Same H formula; CRA may change PQA global/local outputs',
        conditions=['H', 'K_CRA'])
    for path in (args.checkpoint, *[args.checkpoint.parent/n for n in
                 ('campaign_config.json', 'run_status.json', 'training_history.csv', 'source_H_metrics.csv')]):
        plan['input_sha256'][str(path)] = digest(path)
    return plan


def verified_complete(folder, queries):
    state = read_json(folder/'run_status.json')
    if not all(state.get(k) is True for k in ('complete', 'frozen_integrity_passed', 'H_history_parity_passed')) or state.get('processed_queries') != queries:
        raise ValueError('CRA target worker incomplete or failed parity.')
    required = {'metrics_combined.csv', 'metrics_with_reference.csv', 'deltas.csv', 'image_scores.csv',
                'score_checks.csv', 'history_metric_checks.csv', 'references.csv', 'sample_manifest.csv', 'experiment_config.json'}
    if not required <= state.get('output_sha256', {}).keys(): raise ValueError('CRA worker receipt incomplete.')
    for name, value in state['output_sha256'].items():
        if Path(name).name != name or digest(folder/name) != value: raise ValueError('CRA target output changed: '+name)


def worker(args, session, plan):
    import torch
    key = f'{args._dataset}/{args._shot}'
    if key not in plan['jobs']: raise ValueError('Unscheduled CRA worker.')
    output = project_path(args._output, ROOT)
    if output.parent != session/args._dataset: raise ValueError('CRA output escaped dataset directory.')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, dataset=args._dataset, shot=args._shot)
    write_json(output/'run_status.json', status)
    state = hook = None
    try:
        verify_plan(plan)
        inputs = recorded_inputs(args.reference_run, args._dataset, args._shot)
        history, old_scores = verified_history(inputs)
        write_json(output/'experiment_config.json', dict(inputs['config'], experiment='K',
            cra_checkpoint=str(args.checkpoint), cra_checkpoint_sha256=plan['cra_checkpoint_sha256'],
            cra_selected_epoch=plan['cra_epoch'], h_checkpoint=plan['h_checkpoint'],
            trainable_base_modules=False, local_fraction=.125, image_coefficient=.25, align_used=True,
            result_dir=str(output), image_score_source=plan['image_score_source']))
        write_csv(output/'references.csv', list(inputs['reference_paths'][0]), inputs['reference_paths'])
        write_csv(output/'sample_manifest.csv', list(inputs['manifest'][0]), inputs['manifest'])
        state = load_model(Path(plan['h_checkpoint']), inputs, output, args.num_workers)
        cra, hook = attach(state, args.checkpoint)
        cra.eval().requires_grad_(False)
        snapshot = freeze_base([None, cra])
        table = evaluate(state, hook, state.loader, output, expected=inputs['manifest'], history_scores=old_scores,
            zero_init=plan['cra_epoch'] == 0, progress=lambda batch: write_progress(output/'evaluation_progress.json',
                dict(dataset=args._dataset, shot=args._shot, batch=batch)))
        combined, checks, deltas = compare(table, history, inputs['classes'])
        for name, rows in (('metrics_with_reference.csv', combined), ('history_metric_checks.csv', checks), ('deltas.csv', deltas)):
            write_csv(output/name, list(rows[0]), rows)
        if not all(r['Passed'] for r in checks): raise ValueError('CRA-off H six metrics differ from historical H.')
        for frozen in (*state.snapshots, snapshot): check_frozen(frozen)
        verify_plan(plan)
        files = [p for p in output.iterdir() if p.is_file() and p.name not in ('run_status.json', 'evaluation_progress.json')]
        status.update(complete=True, processed_queries=len(inputs['manifest']), frozen_integrity_passed=True,
            H_history_parity_passed=True, historical_full_pixel_arrays_compared=False,
            cra_selected_epoch=plan['cra_epoch'], output_sha256={p.name:digest(p) for p in files})
        write_json(output/'run_status.json', status)
    except BaseException as error:
        write_json(output/'run_status.json', dict(status, complete=False, error=str(error) or 'KeyboardInterrupt'))
        raise
    finally:
        if hook is not None: hook.close()
        state = None; cleanup()


def summarize(session, status):
    metrics, deltas = [], []
    for key, path in status['job_results'].items():
        dataset, shot = key.split('/')
        metrics.extend(dict(Dataset=dataset, Shot=int(shot), **r) for r in read_csv(Path(path)/'metrics_with_reference.csv'))
        deltas.extend(dict(Dataset=dataset, Shot=int(shot), **r) for r in read_csv(Path(path)/'deltas.csv'))
    for name, rows in (('metrics_summary.csv', [r for r in metrics if r['Category'] == 'Mean']),
        ('metrics_by_category.csv', [r for r in metrics if r['Category'] != 'Mean']), ('deltas_by_category.csv', deltas),
        ('deltas_vs_H.csv', [r for r in deltas if r['Comparator'] == 'H' and r['Category'] == 'Mean']),
        ('deltas_vs_original.csv', [r for r in deltas if r['Comparator'] == 'baseline' and r['Category'] == 'Mean'])):
        write_csv(session/name, list(rows[0]), rows)
    records, overall = retention(deltas)
    write_csv(session/'retention_by_shot.csv', list(records[0]), records)
    write_json(session/'verification_summary.json', dict(experiment='K', H_history_parity_passed=True,
        image_score_formula_unchanged=True, candidate_scores_may_change=True, retention=overall))


def main(argv=None):
    args = parse_args(argv)
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    if args.status:
        session = project_path(args.status, ROOT); session.relative_to(RESULT_ROOT.resolve())
        status = read_json(session/'run_status.json')
        print('只读最后保存进度，不启动模型。'); print('当前数据集/shot：', status.get('current'))
        print('已完成：', status.get('completed_jobs', [])); return
    if args._session:
        session = project_path(args._session, ROOT); session.relative_to(RESULT_ROOT.resolve())
        plan = read_json(session/'campaign_config.json')
        for k,v in plan['options'].items(): setattr(args, k, project_path(v, ROOT) if k in ('checkpoint', 'reference_run') else v)
        worker(args, session, plan); return
    if args.resume_from:
        session = project_path(args.resume_from, ROOT); session.relative_to(RESULT_ROOT.resolve())
        plan = read_json(session/'campaign_config.json'); verify_plan(plan)
        for k,v in plan['options'].items(): setattr(args, k, project_path(v, ROOT) if k in ('checkpoint', 'reference_run') else v)
        status = read_json(session/'run_status.json')
        if set(status['job_results'])-set(plan['jobs']): raise ValueError('CRA resume worker plan differs.')
    else:
        plan = preflight(args)
        session = new_directory(RESULT_ROOT, 'K_seed10')
        write_json(session/'campaign_config.json', plan)
        status = dict(complete=False, current=None, current_output=None, completed_jobs=[], job_results={})
        write_json(session/'run_status.json', status)
    if receipt: write_json(receipt, dict(session=str(session), phase='testing'))
    try:
        for key, spec in plan['jobs'].items():
            dataset, shot = key.split('/'); previous = status['job_results'].get(key)
            status.update(complete=False, current=key, current_output=previous)
            if previous:
                old = Path(previous); old.resolve().relative_to(session)
                try: done = read_json(old/'run_status.json')
                except (OSError, ValueError): done = {}
                if done.get('complete'):
                    verified_complete(old, spec['queries'])
                    if key not in status['completed_jobs']: status['completed_jobs'].append(key)
                    continue
            parent = session/dataset; parent.mkdir(parents=True, exist_ok=True)
            output = parent/f'{shot}shot_seed10'; attempt = 2
            while output.exists(): output = parent/f'{shot}shot_seed10_run{attempt:02d}'; attempt += 1
            status.update(current_output=str(output)); status.pop('error', None)
            status['job_results'][key] = str(output)
            if key in status['completed_jobs']: status['completed_jobs'].remove(key)
            write_json(session/'run_status.json', status)
            print(f'CRA测试：数据集={dataset}，shot={shot}', flush=True)
            run_child([sys.executable, '-u', str(Path(__file__).resolve()), '--_session', str(session),
                '--_output', str(output), '--_dataset', dataset, '--_shot', shot])
            verified_complete(output, spec['queries']); cleanup()
            status['completed_jobs'].append(key); write_json(session/'run_status.json', status)
        summarize(session, status); verify_plan(plan)
        status.update(complete=True, current=None, current_output=None); status.pop('error', None)
        write_json(session/'run_status.json', status)
        if receipt: write_json(receipt, dict(session=str(session), phase='completed'))
        print('CRA18条件测试完成：', session, flush=True)
    except BaseException as error:
        status.update(complete=False, error=str(error) or 'KeyboardInterrupt'); write_json(session/'run_status.json', status)
        print(f'CRA测试停止：数据集/shot={status["current"]}，后续不启动。\n{session}', flush=True)
        raise


if __name__ == '__main__': main()
