r"""实验J：H同权重ALIGN定位对照，不训练、不选参。

Root: python .\models_test\rma_align_ablation.py
models_test: python .\rma_align_ablation.py
默认六数据集，每集1/2/4-shot、seed10，每条件独立子进程并等待退出清理。
两组共用实时计算的H图像分数；仅12.5%定位混合图是否经过ALIGN不同。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, write_csv
from h_inference import WEIGHT_SHA, recorded_inputs
from rma_align_ablation_utils import CONDITIONS, DATASETS, comparison_tables, retention, verified_history
from rma_control_io import write_json, write_progress
from rma_utils import cleanup, digest, run_child

RESULT_ROOT = ROOT/'results'/Path(__file__).stem
REQUIRED_FLAGS = ('complete', 'frozen_integrity_passed', 'history_H_parity_passed',
                  'fresh_H_pixel_parity_passed', 'image_scores_identical', 'image_metrics_identical')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', default='results/train_rma_scratch_new/mvtec_seed10/best.pth')
    p.add_argument('--reference_run', default='results/rma_score_coordination_inference/experiment_H_inference_seed10')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    p.add_argument('--status', help='Read saved dataset/shot progress without starting a model.')
    p.add_argument('--resume_from', help='Reuse completed checked conditions; rerun an incomplete condition in a fresh folder.')
    for name in ('session', 'output', 'dataset', 'shot'):
        p.add_argument('--_'+name, help=argparse.SUPPRESS, type=int if name == 'shot' else str)
    args = p.parse_args(argv)
    if (args.datasets != sorted(set(args.datasets), key=DATASETS.index)
            or args.shots != sorted(set(args.shots)) or args.num_workers < 0):
        p.error('Datasets/shots must be unique and in the default order; workers must be nonnegative.')
    hidden = [getattr(args, '_'+n) is not None for n in ('session', 'output', 'dataset', 'shot')]
    if any(hidden) and not all(hidden):
        p.error('Incomplete internal worker fields.')
    tokens = sys.argv[1:] if argv is None else argv
    if args.status and args.resume_from:
        p.error('Use --status or --resume_from separately.')
    if args.resume_from and any(t.startswith('--') and t.split('=', 1)[0] != '--resume_from' for t in tokens):
        p.error('Resume restores saved options; use --resume_from alone.')
    if any(hidden) and (args.status or args.resume_from):
        p.error('Workers cannot use status/resume mode.')
    for name in ('checkpoint', 'reference_run'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def code_hashes():
    names = ('rma_align_ablation.py', 'rma_align_ablation_utils.py', 'h_inference.py', 'rma_control_io.py',
        'fusion_ablation.py', 'fusion_ablation_utils.py', 'branch_diagnostics_utils.py',
        'rma_fixed_fusion_utils.py', 'rma_output_ablation_utils.py', 'rma_score_coordination_utils.py',
        'rma_utils.py', 'rma_model.py', 'rma_scratch_common.py', 'rma_joint_utils.py')
    files = [ROOT/'test.py', ROOT/'train.py', *(ROOT/'models_test'/n for n in names)]
    for name in ('adaptcliplib', 'dataset', 'tools', 'metrics'):
        files.extend(sorted((ROOT/name).rglob('*.py')))
    return {str(p): digest(p) for p in files}


def verify_files(config):
    if config['code_sha256'] != code_hashes():
        raise ValueError('Inference/evaluation code changed since this campaign started.')
    for path, expected in config['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Weight, metadata or historical input changed: {path}')


def preflight(args):
    """Standard-library input audit only; no model/torch import in the parent."""
    if digest(args.checkpoint) != WEIGHT_SHA:
        raise ValueError('Requires the fixed new-D epoch14 checkpoint used by H.')
    files = {str(args.checkpoint): WEIGHT_SHA}
    manifest = args.checkpoint.parent/'model_manifest.json'
    files[str(manifest)] = digest(manifest)
    jobs = {}
    for dataset in args.datasets:
        for shot in args.shots:
            inputs = recorded_inputs(args.reference_run, dataset, shot)
            verified_history(inputs)
            paths = [inputs['meta'], Path(inputs['config']['pretrained_model'])]
            paths += [inputs['history']/n for n in ('experiment_config.json', 'run_status.json',
                'input_audit.json', 'references.csv', 'sample_manifest.csv', 'image_scores.csv', 'metrics_combined.csv')]
            for path in paths:
                if str(path) not in files:
                    files[str(path)] = digest(path)
            jobs[f'{dataset}/{shot}'] = dict(queries=len(inputs['manifest']), classes=inputs['classes'])
    opts = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            if not k.startswith('_') and k not in ('status', 'resume_from')}
    return dict(experiment='J', options=opts, jobs=jobs, code_sha256=code_hashes(), input_sha256=files,
        epoch=14, seed=10, image_coefficient=.25, local_fraction=.125, sigma=4,
        conditions=CONDITIONS, image_score_source='Fresh H for BOTH conditions; SE keeps G0_L1 25% scoring path',
        baseline_source='Historical original A verified by H; not freshly inferred in J')


def restore_options(args, config):
    for name, value in config['options'].items():
        setattr(args, name, project_path(value, ROOT) if name in ('checkpoint', 'reference_run') else value)


def worker(args, session, config):
    import math
    import torch
    from h_inference import load_model
    from rma_align_ablation_utils import predict
    from rma_utils import MetricsBuffer, check_frozen, save_tables
    from tqdm import tqdm
    key = f'{args._dataset}/{args._shot}'
    if key not in config['jobs']:
        raise ValueError('Worker condition is absent from the campaign.')
    output = project_path(args._output, ROOT)
    if output.parent != session/args._dataset:
        raise ValueError('Worker output must stay inside its own dataset directory.')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, phase='initializing', dataset=args._dataset, shot=args._shot, processed_queries=0)
    write_json(output/'run_status.json', status)
    buffer = state = None
    try:
        verify_files(config)
        inputs = recorded_inputs(args.reference_run, args._dataset, args._shot)
        history, historical_scores = verified_history(inputs)
        write_json(output/'experiment_config.json', dict(inputs['config'], experiment='J',
            result_dir=str(output), checkpoint=str(args.checkpoint), epoch=14, seed=10,
            image_coefficient=.25, local_fraction=.125, sigma=4, conditions=CONDITIONS,
            align_used={'H': True, 'J_no_ALIGN': False}, image_score_source=config['image_score_source']))
        write_csv(output/'references.csv', list(inputs['reference_paths'][0]), inputs['reference_paths'])
        write_csv(output/'sample_manifest.csv', list(inputs['manifest'][0]), inputs['manifest'])
        write_json(output/'input_audit.json', read_json(inputs['history']/'input_audit.json'))
        state = load_model(args.checkpoint, inputs, output, args.num_workers)
        pixels, images, scores, score_checks = [], [], [], []
        closed, count, batches = set(), 0, 0

        def flush():
            p, i = buffer.finish(state.device)
            pixels.extend(p); images.extend(i)
            save_tables(output, pixels, images)
            cleanup()

        with torch.no_grad():
            for items in tqdm(state.loader, desc=f'J ALIGN {args._dataset} {args._shot}-shot'):
                maps, chosen, parts = predict(state, items['img'], items['cls_name'], items['img_path'])
                if not torch.equal(chosen['H'], chosen['J_no_ALIGN']):
                    raise ValueError('ALIGN toggle changed image scores.')
                if any(m.shape != items['img_mask'][:, 0].shape for m in maps.values()):
                    raise ValueError('Local map/ground-truth shape differs.')
                if chosen['H'].shape != (len(items['cls_name']),):
                    raise ValueError('Image score shape differs.')
                for i, category in enumerate(items['cls_name']):
                    if count >= len(inputs['manifest']):
                        raise ValueError('Unexpected extra query.')
                    path, label = items['img_path'][i], int(items['anomaly'][i])
                    expected = inputs['manifest'][count]
                    if (category != expected['Category'] or path_key(path) != path_key(expected['Image Path'])
                            or label != int(expected['Sample Type'] == 'Anomaly')):
                        raise ValueError('Query order/identity/label differs from H.')
                    if buffer is None or category != buffer.category:
                        if buffer is not None:
                            flush(); closed.add(buffer.category)
                        if category in closed:
                            raise ValueError('Noncontiguous category; cannot compute full category metrics.')
                        buffer = MetricsBuffer(category, conditions=CONDITIONS)
                    buffer.append(items['img_mask'][i, 0] > .5, label, maps, chosen, i)
                    delta = float(chosen['H'][i])-float(historical_scores[count]['H Image Score'])
                    check = dict(Category=category, **{'Image Path': path}, Delta=delta,
                                 Passed=math.isfinite(delta) and abs(delta) <= 1e-6)
                    score_checks.append(check)
                    if not check['Passed']:
                        write_csv(output/'score_checks.csv', list(check), score_checks)
                        raise ValueError('Fresh H image score disagrees with verified history.')
                    scores.append(dict(Category=category, **{'Image Path': path}, Label=label,
                        H=float(chosen['H'][i]), J_no_ALIGN=float(chosen['J_no_ALIGN'][i]),
                        SE=float(parts['E'][i]), G0=float(parts['G0'][i])))
                    count += 1
                batches += 1
                status.update(phase='evaluating', processed_queries=count)
                write_progress(output/'evaluation_progress.json', status)
                del maps, chosen, parts
        if buffer is None or count != config['jobs'][key]['queries']:
            raise ValueError('Incomplete test query coverage.')
        flush()
        table = save_tables(output, pixels, images)
        combined, parity, identity, deltas = comparison_tables(table, history, inputs['classes'])
        for name, rows in (('image_scores.csv', scores), ('score_checks.csv', score_checks),
                ('history_metric_checks.csv', parity), ('image_metric_identity_checks.csv', identity),
                ('metrics_with_reference.csv', combined), ('deltas.csv', deltas)):
            write_csv(output/name, list(rows[0]), rows)
        if not all(r['Passed'] for r in (*parity, *identity)):
            raise ValueError('H endpoint six-metric parity or identical image metrics failed.')
        for snapshot in state.snapshots:
            check_frozen(snapshot)
        verify_files(config)
        files = [p for p in output.iterdir() if p.is_file() and p.name not in
                 ('run_status.json', 'evaluation_progress.json')]
        status.update(phase='completed', processed_queries=count, query_encoder_calls=batches,
            old_three_adapter_calls=batches, rma_calls=batches,
            **{name: True for name in REQUIRED_FLAGS},
            historical_full_pixel_arrays_compared=False,
            output_sha256={p.name: digest(p) for p in files})
        write_json(output/'run_status.json', status)
    except BaseException as error:
        status.update(complete=False, phase='stopped', error=str(error))
        write_json(output/'run_status.json', status)
        raise
    finally:
        buffer = state = None
        cleanup()


def verified_complete(folder, expected_queries):
    state = read_json(folder/'run_status.json')
    if not all(state.get(k) is True for k in REQUIRED_FLAGS) or state.get('processed_queries') != expected_queries:
        raise ValueError('Worker has not completed all J checks.')
    required = {'metrics_combined.csv', 'metrics_with_reference.csv', 'deltas.csv', 'image_scores.csv',
                'score_checks.csv', 'history_metric_checks.csv', 'image_metric_identity_checks.csv',
                'references.csv', 'sample_manifest.csv', 'input_audit.json', 'experiment_config.json'}
    hashes = state.get('output_sha256', {})
    if not required <= hashes.keys():
        raise ValueError('Worker output receipt is incomplete.')
    for name, value in hashes.items():
        if Path(name).name != name or digest(folder/name) != value:
            raise ValueError('Completed worker output changed: '+name)


def summarize(session, state):
    metrics, deltas = [], []
    for key, directory in state['job_results'].items():
        dataset, shot = key.split('/')
        for name, rows in (('metrics_with_reference.csv', metrics), ('deltas.csv', deltas)):
            rows.extend(dict(Dataset=dataset, Shot=int(shot), **r) for r in read_csv(Path(directory)/name))
    for name, rows in (('metrics_summary.csv', [r for r in metrics if r['Category'] == 'Mean']),
                       ('metrics_by_category.csv', [r for r in metrics if r['Category'] != 'Mean']),
                       ('deltas_by_category.csv', deltas),
                       ('deltas_vs_H.csv', [r for r in deltas if r['Comparator'] == 'H' and r['Category'] == 'Mean']),
                       ('deltas_vs_original.csv', [r for r in deltas if r['Comparator'] == 'baseline' and r['Category'] == 'Mean'])):
        write_csv(session/name, list(rows[0]), rows)
    rows, overall = retention(deltas)
    write_csv(session/'retention_by_shot.csv', list(rows[0]), rows)
    write_json(session/'verification_summary.json', dict(experiment='J',
        all_H_history_checks_passed=True, fresh_H_pixel_parity_passed=True, image_scores_unchanged=True,
        historical_full_pixel_arrays_compared=False, retention=overall))


def show_status(session):
    state = read_json(session/'run_status.json')
    current = state.get('current')
    print('只读取最后保存进度，不启动模型，也不据此判断进程是否仍在运行。')
    print('已完成全部条件。' if state.get('complete') else f'当前测试条件：{current or "尚未启动"}')
    print('已完成：', state.get('completed_jobs', []))
    print('结果目录：', session)
    if state.get('error'):
        print('错误/中断：', state['error'])


def main(argv=None):
    args = parse_args(argv)
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    if args.status:
        session = project_path(args.status, ROOT); session.relative_to(RESULT_ROOT.resolve())
        show_status(session); return
    if args._session:
        session = project_path(args._session, ROOT); session.relative_to(RESULT_ROOT.resolve())
        config = read_json(session/'campaign_config.json'); restore_options(args, config)
        worker(args, session, config); return
    if args.resume_from:
        session = project_path(args.resume_from, ROOT); session.relative_to(RESULT_ROOT.resolve())
        config = read_json(session/'campaign_config.json'); restore_options(args, config)
        verify_files(config)
        state = read_json(session/'run_status.json')
        if set(state['job_results'])-set(config['jobs']):
            raise ValueError('Saved worker plan differs from campaign.')
    else:
        print(f'实验J：固定H权重和图像分数；只切换定位ALIGN。正在核对{len(args.datasets)*len(args.shots)}条件输入。', flush=True)
        config = preflight(args)
        session = new_directory(RESULT_ROOT, 'J_seed10')
        write_json(session/'campaign_config.json', config)
        state = dict(complete=False, current=None, current_output=None, completed_jobs=[], job_results={})
        write_json(session/'run_status.json', state)
    if receipt:
        write_json(receipt, dict(session=str(session), phase='testing'))
    try:
        for key, spec in config['jobs'].items():
            dataset, shot = key.split('/')
            previous = state['job_results'].get(key)
            state.update(complete=False, current=key, current_output=previous)
            if previous:
                old = Path(previous); old.resolve().relative_to(session)
                try:
                    receipt = read_json(old/'run_status.json')
                except (OSError, ValueError):
                    receipt = {}  # Never reuse an unreadable completion receipt.
                if receipt.get('complete') is True:
                    verified_complete(old, spec['queries'])
                    if key not in state['completed_jobs']:
                        state['completed_jobs'].append(key)
                    print('已核对并复用：', key, flush=True); continue
            parent = session/dataset; parent.mkdir(parents=True, exist_ok=True)
            output = parent/f'{shot}shot_seed10'; attempt = 2
            while output.exists():
                output = parent/f'{shot}shot_seed10_run{attempt:02d}'; attempt += 1
            state.update(complete=False, current=key, current_output=str(output))
            state.pop('error', None)
            state['job_results'][key] = str(output)
            if key in state['completed_jobs']:
                state['completed_jobs'].remove(key)
            write_json(session/'run_status.json', state)
            print(f'实验J：数据集={dataset}，shot={shot}；独立子进程', flush=True)
            run_child([sys.executable, '-u', str(Path(__file__).resolve()), '--_session', str(session),
                       '--_output', str(output), '--_dataset', dataset, '--_shot', shot])
            verified_complete(output, spec['queries'])
            cleanup()  # Child has already exited; its live tensors/CUDA context are released.
            state['completed_jobs'].append(key)
            write_json(session/'run_status.json', state)
        summarize(session, state)
        verify_files(config)
        state.update(complete=True, current=None, current_output=None); state.pop('error', None)
        write_json(session/'run_status.json', state)
        if receipt:
            write_json(receipt, dict(session=str(session), phase='completed'))
        print('实验J完成：', session, flush=True)
    except BaseException as error:
        state.update(complete=False, error=str(error))
        write_json(session/'run_status.json', state)
        print(f"运行停止，后续不启动。当前数据集/shot：{state['current']}\n已完成：{state['completed_jobs']}\n结果目录：{session}\n当前子任务：{state['current_output']}\n错误：{error}", flush=True)
        raise


if __name__ == '__main__':
    main()
