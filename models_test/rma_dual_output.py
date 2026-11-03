"""Old B best.pth: original A, complete B, and A-image/B-final-map dual output."""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS, IDS, prepare
from fusion_ablation_utils import new_directory, project_path, read_csv, read_json, write_csv, write_json
from rma_dual_output_utils import CONDITIONS, DualOutputBuffer, check_history, route_outputs
from rma_utils import FIELDS, cleanup, digest, failure_status, run_child, record_pipeline_progress
from test_rma import inference, preflight

OLD_B = 'results/test_rma/pipeline_20261030_210844_733256'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--reference_run', default=OLD_B, help='Completed B session; uses its own source-selected best.pth.')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_maps', action='store_true', help='Save selected sample maps only; default off.')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--_session', help=argparse.SUPPRESS)
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    p.add_argument('--_dataset', choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_workers < 0 or len(set(args.datasets)) != len(args.datasets) or args.shots != sorted(set(args.shots)):
        p.error('Nonnegative workers, unique datasets, ascending unique shots required.')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        p.error('history_session must be a directory name.')
    internal = [args._session is not None, args._dataset is not None, args._shot is not None]
    if any(internal) and not all(internal):
        p.error('All worker arguments must be provided together.')
    for name in ('reference_run', 'visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    campaign = read_json(args.reference_run/'campaign_config.json')
    if read_json(args.reference_run/'run_status.json').get('complete') is not True or campaign.get('disable_rma') is not False:
        p.error('reference_run must be a completed enabled-B evaluation.')
    args.checkpoint = Path(campaign['checkpoint'])
    training_status = read_json(args.checkpoint.parent/'run_status.json')
    if (args.checkpoint.name != 'best.pth' or training_status.get('complete') is not True
            or Path(training_status['best_checkpoint']).resolve() != args.checkpoint.resolve()
            or digest(args.checkpoint) != campaign['checkpoint_sha256']):
        p.error('Historical source-selected best checkpoint is missing, changed, or mismatched.')
    if not set(args.datasets) <= set(campaign['datasets']) or not set(args.shots) <= set(campaign['shots']):
        p.error('Requested scope is absent from the historical B evaluation.')
    args.disable_rma = False
    return args


def summarize(session, jobs):
    """All six metrics and shot-wise retention for B and dual output versus A."""
    all_rows, deltas = [], []
    for dataset, shot in jobs:
        table = read_csv(session/dataset/f'{shot}shot_seed10'/'metrics_combined.csv')
        all_rows.extend(dict(Dataset=dataset, Shot=shot, **r) for r in table)
        means = {r['Condition']: r for r in table if r['Category'] == 'Mean'}
        if set(means) != set(CONDITIONS):
            raise ValueError('Three conditions required.')
        for condition in CONDITIONS[1:]:
            values = {f: float(means[condition][f])-float(means['baseline'][f]) for f in FIELDS}
            deltas.append(dict(Dataset=dataset, Shot=shot, Condition=condition, **values, Net=sum(values.values())))
    for filename, is_mean in [('metrics_summary.csv', True), ('metrics_by_category.csv', False)]:
        write_csv(session/filename, ['Dataset', 'Shot', 'Category', 'Condition', *FIELDS],
                  [r for r in all_rows if (r['Category'] == 'Mean') == is_mean])
    write_csv(session/'deltas_vs_original.csv', ['Dataset', 'Shot', 'Condition', *FIELDS, 'Net'], deltas)
    retention = []
    for shot in sorted({s for _, s in jobs}):
        for condition in CONDITIONS[1:]:
            group = [r for r in deltas if r['Shot'] == shot and r['Condition'] == condition]
            gain, loss = sum(max(r['Net'], 0) for r in group), -sum(min(r['Net'], 0) for r in group)
            positive = sum(r['Net'] > 0 for r in group)
            complete = {r['Dataset'] for r in group} == set(DATASETS)
            retention.append(dict(Shot=shot, Condition=condition, CompletedDatasets=len(group), PositiveDatasets=positive,
                                  Gains=gain, Losses=loss, Net=gain-loss, SixDatasetsComplete=complete,
                                  Retain=bool(complete and positive >= 3 and gain > loss)))
    write_csv(session/'retention_by_shot.csv', list(retention[0]), retention)


def worker(args):
    session = project_path(args._session, ROOT)
    session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    output = session/args._dataset/f'{args._shot}shot_seed10'
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('rma-dual-output')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'test.log', encoding='utf-8')
    logger.addHandler(handler)
    try:
        source, control, audit = prepare(args, args._dataset, args._shot)
        if read_json(session/'input_audit.json')[args._dataset][str(args._shot)] != audit:
            raise ValueError('Historical inputs changed after preflight.')
        checkpoint_hash = digest(args.checkpoint)
        if checkpoint_hash != read_json(session/'campaign_config.json')['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after preflight.')
        historical = args.reference_run/args._dataset/f'{args._shot}shot_seed10'
        history_status = read_json(historical/'run_status.json')
        if not all(history_status.get(k) is True for k in ('complete', 'baseline_parity_passed', 'frozen_integrity_passed')):
            raise ValueError('Historical B round is incomplete or has failed parity checks.')
        write_json(output/'experiment_config.json', dict(**source['config'], rma_checkpoint=str(args.checkpoint),
            rma_sha256=checkpoint_hash, reference_run=str(args.reference_run), conditions=list(CONDITIONS),
            image_output='original A final score', pixel_output='complete B four-branch + ALIGN + smoothing final map',
            save_candidate_maps=args.save_maps, disable_rma=False))
        write_json(output/'input_audit.json', audit)
        write_csv(output/'references.csv', ['Category', 'Image Path'],
                  [dict(Category=c, **{'Image Path': p}) for c, p in source['reference_paths']])
        write_csv(output/'sample_manifest.csv', list(source['manifest'][0]), source['manifest'])
        write_json(output/'run_status.json', dict(status='running', complete=False))
        status = inference(args, source, control, output, logger, checkpoint_hash,
                           output_adapter=route_outputs, buffer_factory=DualOutputBuffer)
        status.update(check_history(output, historical))
        status.update(dataset=IDS[args._dataset], k_shots=args._shot, seed=10,
                      dual_pixel_metrics='reused B evaluation of identical map inputs',
                      dual_image_metrics='reused A evaluation of identical image-score inputs')
        write_json(output/'run_status.json', status)
    except BaseException as error:
        failure_status(output, error)
        logger.exception('Dual-output evaluation stopped.')
        raise
    finally:
        cleanup()
        handler.close()
        logger.removeHandler(handler)


def main(argv=None):
    args = parse_args(argv)
    if args._session:
        worker(args)
        return
    jobs, audits = preflight(args)
    session = new_directory(ROOT/'results'/Path(__file__).stem, 'old_B_six_datasets_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'testing', session=str(session), evaluation_jobs=jobs)
    write_json(session/'input_audit.json', audits)
    write_json(session/'campaign_config.json', dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        reference_run=str(args.reference_run), datasets=args.datasets, shots=args.shots, seed=10,
        conditions=list(CONDITIONS), comparator='original AdaptCLIP', training=False,
        output_rule='A final image score + complete B final pixel map; one common B checkpoint for every shot'))
    done = []
    try:
        for dataset, shot in jobs:
            write_json(session/'run_status.json', dict(status='running', complete=False, completed_jobs=done, current=[dataset, shot]))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--reference_run', str(args.reference_run),
                       '--datasets', dataset, '--shots', str(shot),
                       '--_session', str(session), '--_dataset', dataset, '--_shot', str(shot)]
            for key in ('history_session', 'visa_source_root', 'visa_control_root', 'num_workers'):
                command.extend([f'--{key}', str(getattr(args, key))])
            if args.save_maps:
                command.append('--save_maps')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            print(f'[{len(done)+1}/{len(jobs)}] {dataset} {shot}-shot -> {session}', flush=True)
            run_child(command)
            state = read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json')
            if not all(state.get(k) is True for k in ('complete', 'old_B_parity_passed', 'dual_output_parity_passed')):
                raise RuntimeError('Round incomplete or parity failed.')
            done.append((dataset, shot))
            summarize(session, done)
        write_json(session/'run_status.json', dict(status='completed', complete=True, completed_jobs=done))
        record_pipeline_progress(receipt, 'completed', session=str(session), evaluation_jobs=jobs)
    except BaseException as error:
        failure_status(session, error)
        print(f'Stopped at {dataset} {shot}-shot; completed {len(done)}/{len(jobs)}. Status: {session / "run_status.json"}', flush=True)
        raise
    print(f'Dual-output evaluation completed: {session}', flush=True)


if __name__ == '__main__':
    main()
