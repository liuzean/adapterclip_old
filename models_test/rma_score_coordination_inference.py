"""实验H实现验证：固定lambda=.25、新D epoch14、seed10，完整18轮推理。

每轮真实重算E定位图与H图像分数，核对离线H逐图分数/完整六指标。
不训练、不重新校准、不改变E局部比例；默认不新增矩阵或图片。
"""
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rma_fixed_fusion_utils as fixed
from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json
from rma_reliability import target_worker
from rma_score_coordination import parse_args as offline_args
from rma_score_coordination import target_history, verify, write_json
from rma_score_coordination_inference_utils import CONDITIONS, LOCAL, HBuffer, HRouter, final_check
from rma_score_coordination_utils import save, source_selection
from rma_utils import FIELDS, cleanup, digest, run_child


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--h_run', default='results/rma_score_coordination/experiment_H_seed10')
    extra.add_argument('--save_maps', action='store_true')
    extra.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=False)
    extra.add_argument('--_attempt_session', help=argparse.SUPPRESS)
    opts, rest = extra.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = offline_args(rest)
    args.h_run = project_path(opts.h_run, ROOT)
    args.save_maps, args.save_visualizations = opts.save_maps, opts.save_visualizations
    args._attempt_session = opts._attempt_session
    if args._session and args._phase != 'target':
        raise ValueError('Inference validation only accepts target workers.')
    tokens = sys.argv[1:] if argv is None else argv
    if args.resume_from and any(token.startswith('--') and token.split('=', 1)[0] != '--resume_from' for token in tokens):
        raise ValueError('Resume restores recorded settings; use --resume_from alone.')
    return args


def hashes():
    from rma_new_d_dual_common import hashes as e_hashes
    result = e_hashes()
    for name in ('rma_score_coordination_inference.py', 'rma_score_coordination_inference_utils.py',
                 'rma_score_coordination.py', 'rma_score_coordination_utils.py', 'rma_target_diagnostics_utils.py',
                 'rma_local_diagnostics_utils.py'):
        result[name] = digest(ROOT/'models_test'/name)
    return result


def preflight(args):
    old = read_json(args.h_run/'campaign_config.json'); verify(old)
    state = read_json(args.h_run/'run_status.json'); selected = read_json(args.h_run/'locked_config.json')
    if (not state.get('complete') or selected.get('coefficient') != .25 or selected.get('target_labels_used') is not False
            or old['weight_epoch'] != 14 or old['seed'] != 10 or selected['checkpoint_sha256'] != old['checkpoint_sha256']):
        raise ValueError('Requires completed H and its fixed source-selected lambda=.25, epoch14/seed10.')
    source_tables = {}
    files = dict(old['historical_sha256'])
    for shot in (1, 2, 4):
        folder = Path(state['job_results'][f'source/{shot}'])
        if not read_json(folder/'run_status.json').get('complete'):
            raise ValueError('H source history incomplete.')
        source_tables[shot] = read_csv(folder/'source_image_metrics.csv')
    if source_selection(source_tables)[0] != .25:
        raise ValueError('Recorded coefficient does not match original source selection.')
    for name, expected in selected['source_metrics_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Source selection evidence changed.')
        files[name] = expected
    audits, historical_jobs = {}, {}
    for dataset in args.datasets:
        audits[dataset] = {}
        for shot in args.shots:
            source, audit, expected_scores, _ = target_history(args, dataset, shot, old['checkpoint_sha256'])
            folder = Path(state['job_results'][f'target/{dataset}/{shot}'])
            status = read_json(folder/'run_status.json')
            if not status.get('complete') or status.get('checkpoint_sha256') != old['checkpoint_sha256']:
                raise ValueError('Offline target H incomplete or uses other weights.')
            for name, expected in status['artifacts'].items():
                if digest(folder/name) != expected:
                    raise ValueError('Offline H output changed.')
                files[str(folder/name)] = expected
            if (read_json(folder/'input_audit.json') != audit or read_csv(folder/'sample_manifest.csv') != source['manifest']
                    or [(r['Category'], r['Image Path']) for r in read_csv(folder/'references.csv')] != source['reference_paths']):
                raise ValueError('Offline H references/query identities differ.')
            scores = read_csv(folder/'image_scores.csv')
            if len(scores) != len(expected_scores):
                raise ValueError('Offline score coverage differs.')
            for row, expected in zip(scores, expected_scores):
                if (row['Category'] != expected['Category'] or path_key(row['Image Path']) != path_key(expected['Image Path'])
                        or int(row['Label']) != expected['Label'] or float(row['E']) != expected['E'] or float(row['G0']) != expected['G0']
                        or not math.isclose(float(row['H']), .75*expected['E']+.25*expected['G0'], abs_tol=1e-6, rel_tol=0)):
                    raise ValueError('Offline H score identity or fixed combination differs.')
            audits[dataset][str(shot)] = audit
            historical_jobs[f'{dataset}/{shot}'] = str(folder)
            files[str(folder/'run_status.json')] = digest(folder/'run_status.json')
    for name in ('campaign_config.json', 'locked_config.json', 'run_status.json'):
        files[str(args.h_run/name)] = digest(args.h_run/name)
    return dict(checkpoint=old['checkpoint'], checkpoint_sha256=old['checkpoint_sha256'], seed=10, weight_epoch=14,
                coefficient=.25, rma_fraction=.125, implementation_sha256=hashes(), history_sha256=files,
                h_run=str(args.h_run), historical_jobs=historical_jobs, input_audits=audits,
                training=False, recalibration=False, verification='H real inference vs prior offline H; E pixel path unchanged')


def verify_plan(directory, checkpoint, checkpoint_hash):
    plan = read_json(Path(directory)/'locked_config.json')
    if (plan['checkpoint_sha256'] != checkpoint_hash or digest(checkpoint) != checkpoint_hash
            or plan['weight_epoch'] != 14 or plan['coefficient'] != .25 or plan['rma_fraction'] != .125
            or plan['implementation_sha256'] != hashes()):
        raise ValueError('Locked verification weights/implementation changed.')
    for name, value in plan['history_sha256'].items():
        if digest(name) != value:
            raise ValueError(f'Historical evidence changed: {name}')
    return plan


def worker(args):
    master = project_path(args._session, ROOT)
    master.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    attempt = project_path(args._attempt_session, ROOT) if args._attempt_session else master
    attempt.relative_to(master)
    output = attempt/args._dataset/f'{args._shot}shot_seed10'
    if project_path(args._output, ROOT) != output:
        raise ValueError('Worker output directory differs from scheduled attempt.')
    campaign = read_json(master/'campaign_config.json')
    plan = verify_plan(campaign['calibration'], Path(campaign['checkpoint']), campaign['checkpoint_sha256'])
    previous = Path(plan['historical_jobs'][f'{args._dataset}/{args._shot}'])
    router = HRouter(read_csv(previous/'image_scores.csv'))
    experiment = SimpleNamespace(CANDIDATE=LOCAL, CONDITIONS=CONDITIONS, RULE='H-fixed-quarter-real-inference-v1',
        PARAMETER='rma_fraction', verify_calibration=verify_plan, dispersion_map=fixed.dispersion_map,
        gated_output=router.fuse, OUTPUT_ADAPTER=router.route, BUFFER_FACTORY=HBuffer,
        OUTPUT_DEFINITION=dict(image='0.75 * complete E final score + 0.25 * actual new-D G0',
                               pixel='Alias freshly computed E 12.5% post-ALIGN and post-smoothing map'),
        FINAL_CHECK=lambda out, table: final_check(out, table, router, previous,
            args.e_run/args._dataset/f'{args._shot}shot_seed10'))
    target_worker(args, attempt, Path(campaign['checkpoint']), campaign['checkpoint_sha256'], experiment=experiment)


def summarize(session, state, args):
    combined, deltas, decisions = [], [], []
    for dataset in args.datasets:
        for shot in args.shots:
            rows = read_csv(Path(state['job_results'][f'{dataset}/{shot}'])/'metrics_combined.csv')
            combined.extend(dict(Dataset=dataset, Shot=shot, **row) for row in rows)
            means = {row['Condition']: row for row in rows if row['Category'] == 'Mean'}
            for comparator in ('E', 'baseline', 'G0_L1', 'G1_L1'):
                change = {field: float(means['H'][field])-float(means[comparator][field]) for field in FIELDS}
                deltas.append(dict(Dataset=dataset, Shot=shot, Comparator=comparator, **change, Net=sum(change.values())))
    for comparator in ('E', 'baseline', 'G0_L1', 'G1_L1'):
        for shot in args.shots:
            rows = [row for row in deltas if row['Comparator'] == comparator and row['Shot'] == shot]
            net = sum(row['Net'] for row in rows); positive = sum(row['Net'] > 0 for row in rows)
            decisions.append(dict(Comparator=comparator, Shot=shot, ImprovedDatasets=positive, Net=net,
                                  Pass=positive >= 3 and net > 0 if len(rows) == 6 else None))
    save(session/'metrics_summary.csv', combined); save(session/'dataset_deltas.csv', deltas)
    save(session/'retention_by_shot.csv', decisions)
    all_checks = []
    for key, directory in state['job_results'].items():
        dataset, shot = key.split('/')
        all_checks.extend(dict(Dataset=dataset, Shot=int(shot), **row) for row in read_csv(Path(directory)/'H_offline_metric_checks.csv'))
    save(session/'offline_metric_checks.csv', all_checks)
    eligible = len(args.datasets) == 6 and args.shots == [1, 2, 4]
    passes = sum(row['Pass'] is True for row in decisions if row['Comparator'] == 'E')
    write_json(session/'verification_summary.json', dict(complete=True, H_score_parity=True, H_fresh_E_pixel_identity=True,
        six_metric_parity=True, fixed_coefficient=.25, real_target_inference=True,
        full_pixel_metrics_recomputed=True, historical_full_pixel_arrays_compared=False,
        retention_eligible=eligible, passing_shots_vs_E=passes, retained_vs_E=passes >= 2 if eligible else None))


def options(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            if not key.startswith('_') and key not in ('resume_from', 'status')}


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        session = project_path(args.status, ROOT); state = read_json(session/'run_status.json')
        print(session, state)
        if state.get('current_output') and (Path(state['current_output'])/'run_status.json').exists():
            print(read_json(Path(state['current_output'])/'run_status.json'))
        return
    if args._session:
        worker(args); return
    if args.resume_from:
        session = project_path(args.resume_from, ROOT); session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
        campaign = read_json(session/'campaign_config.json')
        for key, value in campaign['options'].items():
            setattr(args, key, project_path(value, ROOT) if key.endswith('_run') or key.endswith('_root') or key == 'source_history' else value)
        plan = verify_plan(campaign['calibration'], Path(campaign['checkpoint']), campaign['checkpoint_sha256'])
        state = read_json(session/'run_status.json')
    else:
        print('H real inference preflight: fixed lambda=.25, epoch14, seed10; no training or calibration.', flush=True)
        plan = preflight(args)
        session = new_directory(ROOT/'results'/Path(__file__).stem, 'experiment_H_inference_seed10')
        directory = session/'verification_plan'; directory.mkdir()
        write_json(directory/'locked_config.json', plan)
        campaign = dict(plan, calibration=str(directory), calibration_sha256=digest(directory/'locked_config.json'), options=options(args))
        write_json(session/'campaign_config.json', campaign); write_json(session/'input_audit.json', plan['input_audits'])
        state = dict(complete=False, phase='inference', current=None, current_output=None, completed_jobs=[], job_results={})
        write_json(session/'run_status.json', state)
    jobs = [(dataset, shot) for dataset in args.datasets for shot in args.shots]
    try:
        for dataset, shot in jobs:
            key = f'{dataset}/{shot}'; previous = state['job_results'].get(key)
            if previous:
                try:
                    status = read_json(Path(previous)/'run_status.json')
                except (OSError, ValueError):
                    status = {}
                if status.get('complete'):
                    if not all(status.get(flag) is True for flag in ('H_offline_score_parity_passed', 'H_pixel_identity_passed', 'H_six_metric_parity_passed', 'frozen_integrity_passed')):
                        raise ValueError('Completed round lacks successful verification evidence.')
                    for name, expected in status['H_output_sha256'].items():
                        if digest(Path(previous)/name) != expected:
                            raise ValueError('Completed result changed.')
                    if key not in state['completed_jobs']:
                        state['completed_jobs'].append(key)
                    print('Reuse verified complete:', key); continue
            attempt = session
            if (session/dataset/f'{shot}shot_seed10').exists():
                attempt = new_directory(session/'retry_attempts', f'{dataset}_{shot}shot_seed10')
                write_json(attempt/'campaign_config.json', campaign); write_json(attempt/'input_audit.json', plan['input_audits'])
            output = attempt/dataset/f'{shot}shot_seed10'
            state.update(complete=False, phase='inference', current=key, current_output=str(output))
            state['job_results'][key] = str(output); write_json(session/'run_status.json', state)
            command = [sys.executable, str(Path(__file__).resolve()), '--_session', str(session), '--_attempt_session', str(attempt),
                       '--_output', str(output), '--_phase', 'target', '--_dataset', dataset, '--_shot', str(shot)]
            for name, value in campaign['options'].items():
                if name == 'save_visualizations':
                    command.append('--save_visualizations' if value else '--no-save_visualizations')
                elif name == 'save_maps':
                    if value:
                        command.append('--save_maps')
                else:
                    command.extend(['--'+name, *map(str, value)] if isinstance(value, list) else ['--'+name, str(value)])
            write_json(session/'last_command.json', dict(command=command, job=key))
            print(f'[{jobs.index((dataset, shot))+1}/{len(jobs)}] H real inference {key}', flush=True)
            run_child(command)
            if not read_json(output/'run_status.json').get('complete'):
                raise ValueError('Worker exited without full verification completion.')
            cleanup(); state['completed_jobs'].append(key); write_json(session/'run_status.json', state)
        verify_plan(campaign['calibration'], Path(campaign['checkpoint']), campaign['checkpoint_sha256'])
        summarize(session, state, args)
        state.update(complete=True, phase='completed', current=None, current_output=None); state.pop('error', None)
        write_json(session/'run_status.json', state)
        print('H real inference verification completed:', session)
    except BaseException as error:
        state.update(complete=False, error=f'{type(error).__name__}: {error}'); write_json(session/'run_status.json', state)
        print(f"Stopped at {state['current']}; completed={state['completed_jobs']}\nSession={session}\nWorker={state['current_output']}", flush=True)
        raise


if __name__ == '__main__':
    main()
