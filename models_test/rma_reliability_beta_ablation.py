"""实验十七补充：β=4/16/64完整目标消融，β0=G0_L1作共同直接对照。

固定新D第14轮、seed10；六数据集各1/2/4-shot。不训练、不用源域分数淘汰β。
每dataset/shot一个子进程，一次编码/RMA/分散度计算同时得到三个β。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import new_directory, project_path, read_csv, read_json, write_csv, write_json
from new_d_without_rma import reference_checkpoint
from rma_reliability import child_command, parse_args, target_inputs, target_worker
from rma_reliability_utils import dispersion_map, gated_output, implementation_hashes, summarize
from rma_utils import cleanup, digest, failure_status, record_pipeline_progress, run_child

RULE = 'rma-predeclared-beta-target-ablation-v1'
GRID = {'beta_4': 4., 'beta_16': 16., 'beta_64': 64.}
CONDITIONS = ('baseline', 'G0_L1', 'G1_L1', *GRID)


def hashes():
    return dict(implementation_hashes(), rma_reliability_beta_ablation=digest(Path(__file__)))


def verify_plan(directory, checkpoint, checkpoint_hash):
    plan = read_json(Path(directory)/'locked_config.json')
    if (plan['rule'] != RULE or plan['parameter_origin'] != 'predeclared_target_ablation'
            or plan['source_selected'] is not False or plan['candidate_parameters'] != GRID or plan['beta'] != 4.
            or plan['checkpoint_sha256'] != checkpoint_hash or digest(checkpoint) != checkpoint_hash
            or plan['weight_epoch'] != 14 or plan['seed'] != 10 or plan['implementation_sha256'] != hashes()):
        raise ValueError('Target beta plan/checkpoint/implementation changed.')
    if any(digest(p) != h for p, h in plan['historical_metric_sha256'].items()):
        raise ValueError('Historical beta0/control metrics changed after preflight.')
    return plan


def experiment():
    return SimpleNamespace(CANDIDATE='beta_4', CANDIDATE_PARAMETERS=GRID, CONDITIONS=CONDITIONS,
        PARAMETER='beta', RULE=RULE, verify_calibration=verify_plan, dispersion_map=dispersion_map, gated_output=gated_output)


def summarize_grid(session, jobs):
    """Complete six metrics and separate comparisons/retention for EACH beta."""
    combined = []
    for dataset, shot in jobs:
        table = read_csv(session/dataset/f'{shot}shot_seed10/metrics_combined.csv')
        combined.extend(dict(Dataset=dataset, Shot=shot, Beta=0 if r['Condition'] == 'G0_L1' else GRID.get(r['Condition'], ''), **r) for r in table)
    for filename, mean in [('metrics_summary.csv', True), ('metrics_by_category.csv', False)]:
        write_csv(session/filename, list(combined[0]), [r for r in combined if (r['Category'] == 'Mean') == mean])
    all_deltas = []; retention = []; overall = []
    for name, beta in GRID.items():
        out = session/name; out.mkdir(exist_ok=True)
        # Shared summary computes identical retention rules and all3 comparators.
        # It reads the session children, so write in root then preserve each beta.
        summarize(session, jobs, DATASETS, candidate=name)
        for filename in ('deltas_vs_G0_L1.csv', 'deltas_vs_original_A.csv', 'deltas_vs_new_D.csv',
                         'retention_by_shot.csv', 'retention_overall.json'):
            (out/filename).write_bytes((session/filename).read_bytes())
        for filename in ('deltas_vs_G0_L1.csv', 'deltas_vs_original_A.csv', 'deltas_vs_new_D.csv'):
            all_deltas.extend(dict(Beta=beta, **r) for r in read_csv(out/filename))
        retention.extend(dict(Beta=beta, **r) for r in read_csv(out/'retention_by_shot.csv'))
        overall.extend(dict(Beta=beta, **r) for r in read_json(out/'retention_overall.json'))
    write_csv(session/'all_beta_deltas.csv', list(all_deltas[0]), all_deltas)
    write_csv(session/'all_beta_retention_by_shot.csv', list(retention[0]), retention)
    write_json(session/'all_beta_retention_overall.json', overall)
    # Root generic filenames would otherwise misleadingly contain only beta64.
    for filename in ('deltas_vs_G0_L1.csv', 'deltas_vs_original_A.csv', 'deltas_vs_new_D.csv'):
        comparator = {'deltas_vs_G0_L1.csv': 'G0_L1', 'deltas_vs_original_A.csv': 'baseline', 'deltas_vs_new_D.csv': 'G1_L1'}[filename]
        write_csv(session/filename, list(all_deltas[0]), [r for r in all_deltas if r['Comparator'] == comparator])
    write_csv(session/'retention_by_shot.csv', list(retention[0]), retention)
    write_json(session/'retention_overall.json', overall)
    # Restore Beta column after the shared summarizer's generic metrics tables.
    for filename, mean in [('metrics_summary.csv', True), ('metrics_by_category.csv', False)]:
        write_csv(session/filename, list(combined[0]), [r for r in combined if (r['Category'] == 'Mean') == mean])


def main(argv=None):
    args = parse_args(argv, description=__doc__)
    if args.calibration or args.calibrate_only:
        raise ValueError('This entry evaluates all three betas; it does not select a source-optimal beta.')
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    if args._session:
        session = project_path(args._session, ROOT)
        session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
        if args._phase != 'target':
            raise ValueError('Only target evaluation workers are allowed.')
        campaign = read_json(session/'campaign_config.json')
        if campaign['checkpoint_sha256'] != checkpoint_hash or campaign['implementation_sha256'] != hashes():
            raise ValueError('Scheduled weights/code changed.')
        target_worker(args, session, checkpoint, checkpoint_hash, experiment=experiment()); return
    jobs = [(d, s) for d in args.datasets for s in args.shots]; audits = {}; metric_hashes = {}
    for dataset, shot in jobs:
        _, _, audit = target_inputs(args, dataset, shot, checkpoint_hash)
        audits.setdefault(dataset, {})[str(shot)] = audit
        path = args.ablation_run/dataset/f'{shot}shot_seed10/metrics_combined.csv'
        table = read_csv(path)
        if not all(any(r['Condition'] == n and r['Category'] == 'Mean' for r in table) for n in ('baseline', 'G0_L1', 'G1_L1')):
            raise ValueError('Missing beta0/original/full-D control.')
        metric_hashes[str(path)] = digest(path)
    session = new_directory(ROOT/'results'/Path(__file__).stem, 'six_datasets_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'testing', session=str(session))
    plan_dir = session/'evaluation_plan'; plan_dir.mkdir()
    plan = dict(rule=RULE, parameter_origin='predeclared_target_ablation', source_selected=False,
        beta=4., candidate_parameters=GRID, seed=10, weight_epoch=14, checkpoint_sha256=checkpoint_hash,
        implementation_sha256=hashes(), historical_metric_sha256=metric_hashes)
    write_json(plan_dir/'locked_config.json', plan)
    write_json(session/'campaign_config.json', dict(plan, checkpoint=str(checkpoint), datasets=args.datasets, shots=args.shots,
        conditions=CONDITIONS, primary_comparator='G0_L1', calibration=str(plan_dir),
        calibration_sha256=digest(plan_dir/'locked_config.json'), num_workers=args.num_workers,
        save_maps=args.save_maps, save_visualizations=args.save_visualizations,
        history={n: str(getattr(args, n)) for n in ('ablation_run', 'new_d_run', 'off_run', 'history_session', 'visa_source_root', 'visa_control_root')}))
    write_json(session/'input_audit.json', audits)
    done = []; current = None
    try:
        for dataset, shot in jobs:
            current = [dataset, shot]
            write_json(session/'run_status.json', dict(status='testing_all_betas', complete=False,
                current=current, completed_jobs=done, betas=list(GRID.values())))
            print(f'[{len(done)+1}/{len(jobs)}] {dataset} {shot}-shot / beta4、16、64同次推理', flush=True)
            command = child_command(args, session, 'target', shot, dataset)
            command[2] = str(Path(__file__).resolve())
            run_child(command)
            status = read_json(session/dataset/f'{shot}shot_seed10/run_status.json')
            if (not all(status.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed'))
                    or status['candidate_parameters'] != GRID):
                raise ValueError('Incomplete beta grid round.')
            done.append((dataset, shot)); summarize_grid(session, done); cleanup()
        write_json(session/'run_status.json', dict(status='completed', complete=True, target_evaluation_complete=True,
            completed_jobs=done, betas=list(GRID.values()), candidate_conditions=54 if len(jobs) == 18 else len(jobs)*3))
    except BaseException as error:
        failure_status(session, error)
        print(f'停止：当前={current}，三个beta共同完成{len(done)}/{len(jobs)}个dataset/shot；{session}', flush=True); raise
    print(f'Completed beta target ablation: {session}', flush=True)


if __name__ == '__main__':
    main()
