"""Shared evaluation of from-scratch three/four-adapter candidates versus A.

One subprocess per dataset/shot; target labels are never used to select epochs.
The explicit three-control entry selects a distinct training/checkpoint identity.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS, IDS, prepare
from fusion_ablation_utils import new_directory, project_path, read_json, write_csv, write_json
from rma_scratch_common import FORMAT, THREE_FORMAT, validate_scratch, validate_three_scratch
from rma_scratch_resume_utils import copy_round, reusable_datasets
from rma_utils import cleanup, digest, failure_status, run_child, record_pipeline_progress
from test_rma import preflight
from test_rma_joint import inference, summarize


def parse_testing(flow, argv=None, *, three_adapter_control=False):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', required=True, help='original: training-loss best.pth; new: source-validation best.pth.')
    p.add_argument('--resume_test_run', help='Original flow: reuse complete datasets from this interrupted test session, rerun from the first incomplete dataset.')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_maps', action='store_true')
    if three_adapter_control:
        p.add_argument('--new_d_run', default='results/test_rma_scratch_new/six_datasets_seed10',
                       help='Completed new-D evaluation for matched additional comparisons.')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--_session', help=argparse.SUPPRESS)
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    p.add_argument('--_dataset', choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_workers < 0 or len(set(args.datasets)) != len(args.datasets) or args.shots != sorted(set(args.shots)):
        p.error('Unique datasets, ascending unique shots and nonnegative workers required.')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        p.error('history_session must be one directory name.')
    fields = [args._session is not None, args._dataset is not None, args._shot is not None]
    if any(fields) and not all(fields):
        p.error('All worker fields required together.')
    for name in ('checkpoint', 'visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    manifest = read_json(args.checkpoint.parent/'model_manifest.json')
    state = read_json(args.checkpoint.parent/'run_status.json')
    expected_format = THREE_FORMAT if three_adapter_control else FORMAT
    if manifest.get('format') != expected_format or manifest.get('flow') != flow or state.get('complete') is not True:
        p.error('Checkpoint must belong to completed matching scratch training.')
    if args.checkpoint.resolve() != Path(state['selected_checkpoint']).resolve():
        p.error('Use the declared training-loss/or source-validation-selected best.pth.')
    args.flow, args.disable_rma = flow, three_adapter_control
    args.three_adapter_control = three_adapter_control
    if three_adapter_control:
        args.new_d_run = project_path(args.new_d_run, ROOT)
    if args.resume_test_run:
        if flow != 'original' or args._session:
            p.error('resume_test_run is only for the original-flow parent test scheduler.')
        args.resume_test_run = project_path(args.resume_test_run,ROOT)
    return args


def worker(args, entry_stem):
    session = project_path(args._session, ROOT)
    session.relative_to((ROOT/'results'/entry_stem).resolve())
    output = session/args._dataset/f'{args._shot}shot_seed10'
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(entry_stem)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'test.log', encoding='utf-8')
    logger.addHandler(handler)
    try:
        source, control, audit = prepare(args, args._dataset, args._shot)
        if audit != read_json(session/'input_audit.json')[args._dataset][str(args._shot)]:
            raise ValueError('Historical inputs changed after preflight.')
        checkpoint_hash = digest(args.checkpoint)
        if checkpoint_hash != read_json(session/'campaign_config.json')['checkpoint_sha256']:
            raise ValueError('Scratch checkpoint changed after preflight.')
        write_json(output/'experiment_config.json', dict(**source['config'], training_flow=args.flow,
            joint_checkpoint=str(args.checkpoint), joint_sha256=checkpoint_hash,
            disable_rma=args.disable_rma, save_candidate_maps=args.save_maps,
            matched_new_flow_three_adapter_control=args.three_adapter_control))
        write_json(output/'input_audit.json', audit)
        write_csv(output/'references.csv', ['Category','Image Path'],
                  [dict(Category=c, **{'Image Path':p}) for c,p in source['reference_paths']])
        write_csv(output/'sample_manifest.csv', list(source['manifest'][0]), source['manifest'])
        write_json(output/'run_status.json', dict(status='running', complete=False))
        def validator(payload, base_hash, clip_hash, config):
            validate = validate_three_scratch if args.three_adapter_control else validate_scratch
            validate(payload, base_hash, clip_hash, config, args.flow)
        candidate = 'new_D_three_control' if args.three_adapter_control else f'S_{args.flow}'
        state = inference(args, source, control, output, logger, checkpoint_hash,
                          checkpoint_validator=validator, candidate_name=candidate)
        state.update(training_scheme='new_flow_three_adapter_control' if args.three_adapter_control else f'scratch_{args.flow}', flow=args.flow,
                     dataset=IDS[args._dataset], k_shots=args._shot, seed=10,
                     note='Three adapters trained without RMA from the start.' if args.three_adapter_control else
                          'All four adapters initialized fresh; original A remains an independent frozen comparator.')
        write_json(output/'run_status.json', state)
    except BaseException as error:
        failure_status(output, error)
        logger.exception('Scratch evaluation stopped.')
        raise
    finally:
        cleanup()
        handler.close()
        logger.removeHandler(handler)


def testing_main(flow, entry_stem, entry_file, argv=None, *, three_adapter_control=False):
    args = parse_testing(flow, argv, three_adapter_control=three_adapter_control)
    if args._session:
        worker(args, entry_stem)
        return
    jobs, audit = preflight(args)
    if three_adapter_control:
        from new_d_control_utils import check_evaluation_reference
        check_evaluation_reference(args, jobs, audit)
    reused, restart = [], None
    if args.resume_test_run:
        reused, restart = reusable_datasets(args.resume_test_run,args,jobs,audit,ROOT,entry_stem)
    session = new_directory(ROOT/'results'/entry_stem, 'six_datasets_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'testing', session=str(session), flow=flow, evaluation_jobs=jobs)
    write_json(session/'input_audit.json', audit)
    write_json(session/'reuse_manifest.json',dict(previous_test_run=str(args.resume_test_run) if args.resume_test_run else None,
        reused_jobs=reused,restart_from=restart,policy='Reuse only complete dataset prefixes; rerun every shot of the first incomplete dataset.'))
    candidate = 'new_D_three_control' if three_adapter_control else f'S_{flow}'
    write_json(session/'campaign_config.json', dict(flow=flow, candidate=candidate, checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint), datasets=args.datasets, shots=args.shots, seed=10,
        comparator='original A', disable_rma=args.disable_rma, matched_new_flow_three_adapter_control=three_adapter_control,
        new_d_reference=str(args.new_d_run) if three_adapter_control else None,
        attribution='Matched new-D training without RMA from the start' if three_adapter_control else
        'RMA plus matched original training protocol' if flow == 'original' else 'RMA plus training-flow changes; no matched new-flow three-adapter control'))
    done = []
    try:
        for dataset, shot in jobs:
            write_json(session/'run_status.json', dict(status='testing', complete=False, completed_jobs=done, current=[dataset,shot]))
            if (dataset,shot) in reused:
                copy_round(args.resume_test_run,session,dataset,shot)
                done.append((dataset,shot))
                summarize(session,done,candidate)
                print(f'[{len(done)}/{len(jobs)}] 复用完整结果：{dataset} {shot}-shot',flush=True)
                continue
            command = [sys.executable, '-u', str(entry_file), '--checkpoint', str(args.checkpoint),
                       '--_session', str(session), '--_dataset', dataset, '--_shot', str(shot)]
            for key in ('history_session','visa_source_root','visa_control_root','num_workers'):
                command.extend([f'--{key}',str(getattr(args,key))])
            if three_adapter_control:
                command.extend(['--new_d_run', str(args.new_d_run)])
            if args.save_maps:
                command.append('--save_maps')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            print(f'[{len(done)+1}/{len(jobs)}] {flow}: {dataset} {shot}-shot -> {session}', flush=True)
            run_child(command)
            if read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json').get('complete') is not True:
                raise RuntimeError('Incomplete evaluation round.')
            done.append((dataset,shot))
            summarize(session,done,candidate)
        if three_adapter_control:
            from new_d_control_utils import compare_to_new_d
            compare_to_new_d(args, session, jobs, candidate)
        write_json(session/'run_status.json', dict(status='completed', complete=True, flow=flow, completed_jobs=done))
        record_pipeline_progress(receipt, 'completed', session=str(session), flow=flow, evaluation_jobs=jobs)
    except BaseException as error:
        failure_status(session,error)
        print(f'Stopped: {dataset} {shot}-shot; completed {len(done)}/{len(jobs)}; {session}',flush=True)
        raise
    print(f'Scratch evaluation completed: {session}',flush=True)
