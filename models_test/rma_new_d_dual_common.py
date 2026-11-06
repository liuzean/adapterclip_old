"""Shared frozen new-D dual-output verification; no tensor import in parent."""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import rma_fixed_fusion_utils as fixed
from context_alignment import DATASETS
from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, write_csv, write_json
from new_d_without_rma import reference_checkpoint
from rma_reliability import child_command, parse_args, target_inputs, target_worker
from rma_reliability_utils import ReliabilityBuffer, dispersion_map, gated_output, implementation_hashes, summarize
from rma_utils import FIELDS, cleanup, digest, failure_status, record_pipeline_progress, run_child

ROOT = Path(__file__).resolve().parents[1]
LOCAL = 'local_candidate'
DUAL = 'dual_output'
CONDITIONS = ('baseline', 'G0_L1', 'G1_L1', LOCAL, DUAL)
MODES = {
    'fixed125': dict(parameter='rma_fraction', value=.125, historical_condition='G0_L1_fixed',
        history='results/rma_fixed_fusion/all_fractions_seed10', offline='dual_alpha12.5'),
    'beta4': dict(parameter='beta', value=4., historical_condition='beta_4',
        history='results/rma_reliability_beta_ablation/six_datasets_seed10', offline='dual_beta4'),
}
OFFLINE = 'results/run_rma_followup_sequence/three_tasks_seed10/analysis_20261106/derived_dual_metrics_by_category.csv'
DEFINITION = dict(image_score='Exact G0_L1 FINAL image score including its original25% map-max contribution',
    pixel_map='Exact local_candidate FINAL map after ALIGN and original Gaussian smoothing',
    dual_recomputes_image_score=False, training=False)


def route_dual(raw, maps, scores):
    """Alias tensors only; never call fusion/smoothing/max here."""
    raw[DUAL] = raw[LOCAL]
    maps[DUAL] = maps[LOCAL]
    scores[DUAL] = scores['G0_L1']


class DualBuffer(ReliabilityBuffer):
    """Three unique pixel buffers; evaluate original inputs then reuse dual rows."""
    def __init__(self, category, conditions=CONDITIONS):
        super().__init__(category, conditions=tuple(n for n in conditions if n != DUAL))

    def append(self, mask, label, maps, scores, index):
        import torch
        if not torch.equal(raw_map := maps[DUAL][index], maps[LOCAL][index]):
            raise ValueError('Dual pixel output changed from complete local candidate.')
        if not torch.equal(scores[DUAL][index], scores['G0_L1'][index]):
            raise ValueError('Dual image output changed from complete G0_L1 final score.')
        if not raw_map.isfinite().all():
            raise ValueError('Nonfinite dual map.')
        super().append(mask, label, maps, scores, index)

    def finish(self, device):
        pixels, images = super().finish(device)
        pixels.append(dict(next(r for r in pixels if r['Condition'] == LOCAL), Condition=DUAL))
        images.append(dict(next(r for r in images if r['Condition'] == 'G0_L1'), Condition=DUAL))
        return pixels, images


def hashes():
    names = ('rma_new_d_dual_common.py', 'rma_fixed_dual_output.py', 'rma_beta4_dual_output.py', 'rma_fixed_fusion_utils.py')
    return dict(implementation_hashes(), **{n: digest(ROOT/'models_test'/n) for n in names})


def offline_rows(args, mode, dataset, shot):
    spec = MODES[mode]
    return [r for r in read_csv(args.offline_metrics) if r['Candidate'] == spec['offline'] and r['Dataset'] == dataset and int(r['Shot']) == shot]


def audit_local(args, mode, dataset, shot, checkpoint_hash):
    source, _, audit = target_inputs(args, dataset, shot, checkpoint_hash)
    spec = MODES[mode]; folder = args.local_history/dataset/f'{shot}shot_seed10'
    status = read_json(folder/'run_status.json'); config = read_json(folder/'experiment_config.json')
    if (read_json(args.local_history/'run_status.json').get('target_evaluation_complete') is not True
            or not all(status.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed'))
            or config['checkpoint_sha256'] != checkpoint_hash or config['weight_epoch'] != 14
            or config['rma_global_used_by_candidate'] is not False or config['align_used'] is not True
            or read_json(folder/'input_audit.json') != audit
            or read_csv(folder/'sample_manifest.csv') != source['manifest']
            or [(r['Category'], r['Image Path']) for r in read_csv(folder/'references.csv')] != list(source['reference_paths'])):
        raise ValueError('Local-history completion/weight/input/mode mismatch.')
    keys = ('dataset', 'seed', 'k_shots', 'test_data_path', 'pretrained_model', 'checkpoint_path', 'features_list',
            'image_size', 'batch_size', 'sigma', 'n_ctx', 'vl_reduction', 'pq_mid_dim', 'pq_context',
            'visual_learner', 'textual_learner', 'pq_learner', 'fusion_type')
    if any(config[k] != source['config'][k] for k in keys):
        raise ValueError('Local historical evaluation protocol differs.')
    if config['candidate_parameters'].get(spec['historical_condition']) != spec['value']:
        raise ValueError('Local history does not use the requested fraction/beta.')
    table = read_csv(folder/'metrics_combined.csv')
    lookup = {(r['Category'], r['Condition']): r for r in table}
    expected_categories = {*source['classes'], 'Mean'}
    if {c for c, n in lookup if n == spec['historical_condition']} != expected_categories:
        raise ValueError('Incomplete local historical metrics.')
    offline = offline_rows(args, mode, dataset, shot)
    if len(offline) != len(expected_categories) or {r['Category'] for r in offline} != expected_categories:
        raise ValueError('Offline dual table is incomplete or has duplicate categories.')
    for row in offline:
        for f in FIELDS:
            donor = lookup[row['Category'], 'G0_L1' if f.startswith('I-') else spec['historical_condition']]
            if not math.isclose(float(row[f]), float(donor[f]), rel_tol=0., abs_tol=1e-10):
                raise ValueError('Offline combination does not match exact historical donor metrics.')
    return audit


def verify_plan(directory, checkpoint, checkpoint_hash):
    plan = read_json(Path(directory)/'locked_config.json'); mode = plan['mode']; spec = MODES[mode]
    if (plan['parameter_origin'] != 'predeclared_dual_output_verification' or plan['source_selected'] is not False
            or plan['checkpoint_sha256'] != checkpoint_hash or digest(checkpoint) != checkpoint_hash
            or plan['seed'] != 10 or plan['weight_epoch'] != 14 or plan['implementation_sha256'] != hashes()
            or plan[spec['parameter']] != spec['value'] or plan['output_definition'] != DEFINITION):
        raise ValueError('Dual plan/code/checkpoint changed.')
    if any(digest(path) != value for path, value in plan['history_sha256'].items()):
        raise ValueError('Historical inputs/offline metrics changed after preflight.')
    return plan


def final_check(output, table, args, mode):
    """Check fresh local outputs, exact copied scores and all six offline metrics."""
    dataset = args._dataset; shot = args._shot; spec = MODES[mode]
    folder = args.local_history/dataset/f'{shot}shot_seed10'
    old_scores = read_csv(folder/'image_scores.csv'); fresh = read_csv(output/'image_scores.csv')
    if len(old_scores) != len(fresh):
        raise ValueError('Incomplete per-image dual/local history coverage.')
    score_checks = []
    for current, old in zip(fresh, old_scores):
        if (current['Category'] != old['Category'] or path_key(current['Image Path']) != path_key(old['Image Path']) or current['Label'] != old['Label']):
            raise ValueError('Local historical image identity/order differs.')
        for name, historical in [('G0_L1', 'G0_L1'), (LOCAL, spec['historical_condition']), (DUAL, 'G0_L1')]:
            a, b = float(current[f'{name} Image Score']), float(old[f'{historical} Image Score'])
            score_checks.append(dict(Image=current['Image Path'], Condition=name, Delta=a-b,
                Passed=math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-5)))
        if float(current[f'{DUAL} Image Score']) != float(current['G0_L1 Image Score']):
            raise ValueError('Dual final score was recomputed or otherwise changed.')
    write_csv(output/'dual_historical_score_checks.csv', list(score_checks[0]), score_checks)
    if not all(r['Passed'] for r in score_checks):
        raise ValueError('Fresh candidate/G0_L1 scores differ from audited history.')
    lookup = {(r['Category'], r['Condition']): r for r in table}
    previous = {(r['Category'], r['Condition']): r for r in read_csv(folder/'metrics_combined.csv')}
    checks = []
    for offline in offline_rows(args, mode, dataset, shot):
        category = offline['Category']
        for f in FIELDS:
            now = float(lookup[category, DUAL][f]); expected = float(offline[f])
            local = float(lookup[category, LOCAL][f]); old_local = float(previous[category, spec['historical_condition']][f])
            checks.append(dict(Category=category, Condition=DUAL, Metric=f, Delta=now-expected, Passed=abs(now-expected) <= .001))
            checks.append(dict(Category=category, Condition=LOCAL, Metric=f, Delta=local-old_local, Passed=abs(local-old_local) <= .001))
            donor = 'G0_L1' if f.startswith('I-') else LOCAL
            if now != float(lookup[category, donor][f]):
                raise ValueError('Dual metric does not equal its exact current donor metric.')
    write_csv(output/'offline_dual_metric_checks.csv', list(checks[0]), checks)
    if not all(r['Passed'] for r in checks):
        raise ValueError('Dual offline metrics or local historical metrics disagree.')
    return dict(dual_output_parity_passed=True, offline_combination_parity_passed=True,
        dual_image_score_source='G0_L1 complete final score', dual_pixel_source='local_candidate complete final map')


def main(mode, entry, argv=None):
    spec = MODES[mode]
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--local_history', default=spec['history'])
    extra.add_argument('--offline_metrics', default=OFFLINE)
    import sys
    options, remaining = extra.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = parse_args(remaining, description=f'{mode}: G0_L1 FINAL image score + complete candidate FINAL pixel map; no retraining.')
    args.local_history = project_path(options.local_history, ROOT); args.offline_metrics = project_path(options.offline_metrics, ROOT)
    if args.calibration or args.calibrate_only:
        raise ValueError('Dual verification does not run source calibration or select parameters.')
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    if args._session:
        session = project_path(args._session, ROOT); session.relative_to((ROOT/'results'/entry.stem).resolve())
        campaign = read_json(session/'campaign_config.json')
        if args._phase != 'target' or campaign['mode'] != mode or campaign['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('Wrong dual worker/mode/checkpoint.')
        audit_local(args, mode, args._dataset, args._shot, checkpoint_hash)
        experiment = SimpleNamespace(CANDIDATE=LOCAL, CONDITIONS=CONDITIONS, RULE=f'new-d-{mode}-dual-v1',
            PARAMETER=spec['parameter'], verify_calibration=verify_plan,
            dispersion_map=fixed.dispersion_map if mode == 'fixed125' else dispersion_map,
            gated_output=fixed.gated_output if mode == 'fixed125' else gated_output,
            OUTPUT_ADAPTER=route_dual, BUFFER_FACTORY=DualBuffer, OUTPUT_DEFINITION=DEFINITION,
            FINAL_CHECK=lambda out, table: final_check(out, table, args, mode))
        target_worker(args, session, checkpoint, checkpoint_hash, experiment=experiment); return
    jobs = [(d, s) for d in args.datasets for s in args.shots]; audits = {}; history_hashes = {str(args.offline_metrics): digest(args.offline_metrics)}
    for dataset, shot in jobs:
        audit = audit_local(args, mode, dataset, shot, checkpoint_hash); audits.setdefault(dataset, {})[str(shot)] = audit
        for base in (args.local_history, args.ablation_run):
            for name in ('metrics_combined.csv', 'image_scores.csv', 'input_audit.json', 'references.csv', 'sample_manifest.csv', 'experiment_config.json'):
                path = base/dataset/f'{shot}shot_seed10'/name; history_hashes[str(path)] = digest(path)
    session = new_directory(ROOT/'results'/entry.stem, 'six_datasets_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'testing', session=str(session))
    plan_dir = session/'evaluation_plan'; plan_dir.mkdir()
    plan = dict(mode=mode, parameter_origin='predeclared_dual_output_verification', source_selected=False,
        seed=10, weight_epoch=14, checkpoint_sha256=checkpoint_hash, implementation_sha256=hashes(),
        history_sha256=history_hashes, output_definition=DEFINITION, **{spec['parameter']: spec['value']})
    write_json(plan_dir/'locked_config.json', plan)
    write_json(session/'campaign_config.json', dict(plan, checkpoint=str(checkpoint), local_history=str(args.local_history),
        offline_metrics=str(args.offline_metrics), datasets=args.datasets, shots=args.shots, conditions=CONDITIONS,
        calibration=str(plan_dir), calibration_sha256=digest(plan_dir/'locked_config.json'), primary_comparator='G0_L1',
        num_workers=args.num_workers, save_maps=args.save_maps, save_visualizations=args.save_visualizations))
    write_json(session/'input_audit.json', audits)
    done = []; current = None
    try:
        for dataset, shot in jobs:
            current = [dataset, shot]
            write_json(session/'run_status.json', dict(status='testing_dual', complete=False, mode=mode, current=current, completed_jobs=done))
            print(f'[{len(done)+1}/{len(jobs)}] {mode}双输出 {dataset} {shot}-shot', flush=True)
            command = child_command(args, session, 'target', shot, dataset); command[2] = str(entry)
            command.extend(['--local_history', str(args.local_history), '--offline_metrics', str(args.offline_metrics)])
            run_child(command)
            status = read_json(session/dataset/f'{shot}shot_seed10/run_status.json')
            if not all(status.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed',
                'dual_output_parity_passed', 'offline_combination_parity_passed')):
                raise ValueError('Incomplete dual/parity checks.')
            done.append((dataset, shot)); summarize(session, done, DATASETS, candidate=DUAL); cleanup()
        write_json(session/'run_status.json', dict(status='completed', complete=True, mode=mode, completed_jobs=done,
            target_evaluation_complete=True, dual_output_parity_passed=True, offline_combination_parity_passed=True))
        record_pipeline_progress(receipt, 'completed', session=str(session))
    except BaseException as error:
        failure_status(session, error)
        print(f'停止：{mode}双输出，当前={current}，完成{len(done)}/{len(jobs)}；{session}', flush=True); raise
    print(f'Completed {mode} dual verification: {session}', flush=True)
