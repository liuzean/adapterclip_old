"""Complete target-set fraction ablation; explicitly NOT source-only selection.

Reuse audited 0/25/37.5% histories; only run missing12.5% at seed10/epoch14.
No model training, no dataset-specific switches, no hidden parameter search.
"""
from pathlib import Path
from types import SimpleNamespace

import rma_fixed_fusion_utils as fixed
from context_alignment import DATASETS
from fusion_ablation_utils import new_directory, project_path, read_csv, read_json, write_csv, write_json
from new_d_without_rma import reference_checkpoint
from rma_reliability import target_inputs, target_worker
from rma_reliability_utils import summarize
from rma_utils import FIELDS, cleanup, digest, failure_status, record_pipeline_progress, run_child

ROOT = Path(__file__).resolve().parents[1]
RULE = 'rma-predeclared-four-fraction-target-ablation-v1'


def hashes():
    return dict(fixed.implementation_hashes(), rma_fixed_fusion_grid=digest(Path(__file__)))


def verify_plan(directory, checkpoint, checkpoint_hash):
    """Verify a predeclared target-evaluation plan; never claim source selection."""
    plan = read_json(Path(directory)/'locked_config.json')
    if (plan['rule'] != RULE or plan['parameter_origin'] != 'predeclared_target_ablation'
            or plan['source_selected'] is not False or plan['rma_fraction'] != .125
            or plan['seed'] != 10 or plan['weight_epoch'] != 14
            or plan['checkpoint_sha256'] != checkpoint_hash or digest(checkpoint) != checkpoint_hash
            or plan['implementation_sha256'] != hashes()):
        raise ValueError('Target ablation plan/weights/implementation changed.')
    if any(digest(path) != value for path, value in plan['reused_files_sha256'].items()):
        raise ValueError('Historical metrics changed after reuse audit.')
    return plan


def history(args, fixed_history, dataset, shot, checkpoint_hash):
    """Read-only audit before reusing original conditions; preserve all categories."""
    source, _, audit = target_inputs(args, dataset, shot, checkpoint_hash)
    relative = Path(dataset)/f'{shot}shot_seed10'
    ablation = args.ablation_run/relative; directory = fixed_history/relative
    campaign = read_json(fixed_history/'campaign_config.json')
    config = read_json(directory/'experiment_config.json')
    state = read_json(directory/'run_status.json')
    if (read_json(fixed_history/'run_status.json').get('target_evaluation_complete') is not True
            or campaign['checkpoint_sha256'] != checkpoint_hash or campaign['rma_fraction'] != .375
            or not all(state.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed'))
            or state['rma_fraction'] != .375 or config['weight_epoch'] != 14 or config['rma_fraction'] != .375
            or config['rma_global_used_by_candidate'] is not False or config['align_used'] is not True
            or config['checkpoint_sha256'] != checkpoint_hash
            or read_json(directory/'input_audit.json') != audit
            or read_csv(directory/'references.csv') != read_csv(ablation/'references.csv')
            or read_csv(directory/'sample_manifest.csv') != source['manifest']):
        raise ValueError('37.5% historical run has different inputs/protocol/weights or is incomplete.')
    for key in ('dataset', 'seed', 'k_shots', 'test_data_path', 'pretrained_model', 'checkpoint_path',
                'features_list', 'image_size', 'batch_size', 'sigma', 'n_ctx', 'vl_reduction',
                'pq_mid_dim', 'pq_context', 'visual_learner', 'textual_learner', 'pq_learner', 'fusion_type'):
        if config[key] != source['config'][key]:
            raise ValueError(f'Historical protocol mismatch: {key}')
    a = read_csv(ablation/'metrics_combined.csv'); b = read_csv(directory/'metrics_combined.csv')
    base = {(r['Category'], r['Condition']): r for r in a}
    for row in b:
        if row['Condition'] in ('baseline', 'G0_L1', 'G1_L1'):
            if any(abs(float(row[f])-float(base[row['Category'], row['Condition']][f])) > .001 for f in FIELDS):
                raise ValueError('37.5% historical comparator differs.')
    sources = [(0., 'G0_L0', ablation, a), (.25, 'G0_L1', ablation, a), (.375, fixed.CANDIDATE, directory, b)]
    result = []
    for fraction, condition, origin, table in sources:
        rows = [r for r in table if r['Condition'] == condition]
        if {r['Category'] for r in rows} != {*source['classes'], 'Mean'} or len(rows) != len(source['classes'])+1:
            raise ValueError('Historical category coverage differs.')
        result.extend(dict(Dataset=dataset, Shot=shot, Fraction=fraction, HistoricalCondition=condition,
            Origin=str(origin), **r) for r in rows)
    return audit, result


def merge(session, reused, jobs):
    combined = list(reused)
    for dataset, shot in jobs:
        directory = session/dataset/f'{shot}shot_seed10'
        combined.extend(dict(Dataset=dataset, Shot=shot, Fraction=.125, HistoricalCondition=fixed.CANDIDATE,
            Origin=str(directory), **r) for r in read_csv(directory/'metrics_combined.csv') if r['Condition'] == fixed.CANDIDATE)
    # Make target-grid conditions explicit, without changing any source files.
    for r in combined:
        r['Condition'] = 'fraction_'+str(round(float(r['Fraction'])*1000))
    for name, mean in [('all_fraction_metrics.csv', True), ('all_fraction_metrics_by_category.csv', False)]:
        group = [r for r in combined if (r['Category'] == 'Mean') == mean]
        write_csv(session/name, list(group[0]), group)
    deltas = []; retention = []; overall = []
    means = {(r['Dataset'], int(r['Shot']), float(r['Fraction'])): r for r in combined if r['Category'] == 'Mean'}
    for (dataset, shot, fraction), row in means.items():
        baseline = means[dataset, shot, .25]
        values = {f: float(row[f])-float(baseline[f]) for f in FIELDS}
        deltas.append(dict(Dataset=dataset, Shot=shot, Fraction=fraction, **values, Net=sum(values.values())))
    write_csv(session/'all_fraction_deltas_vs_25.csv', list(deltas[0]), deltas)
    for fraction in (0., .125, .25, .375):
        passed = 0; complete_all = True
        for shot in (1, 2, 4):
            group = [r for r in deltas if r['Shot'] == shot and r['Fraction'] == fraction]
            complete = {r['Dataset'] for r in group} == set(DATASETS)
            positive = sum(r['Net'] > 0 for r in group); net = sum(r['Net'] for r in group)
            ok = complete and positive >= 3 and net > 0
            passed += ok; complete_all &= complete
            retention.append(dict(Fraction=fraction, Shot=shot, PositiveDatasets=positive, Net=net, Complete=complete, Pass=ok))
        overall.append(dict(Fraction=fraction, Comparator=.25, All18Complete=complete_all, PassingShots=passed,
            RetainIncrement=bool(fraction != .25 and complete_all and passed >= 2), ExploratoryTargetAblation=True))
    write_csv(session/'all_fraction_retention_by_shot.csv', list(retention[0]), retention)
    write_json(session/'all_fraction_retention_overall.json', overall)


def main(args, fixed_history):
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    if args._session:
        session = project_path(args._session, ROOT)
        session.relative_to((ROOT/'results/rma_fixed_fusion').resolve())
        if args._phase != 'target':
            raise ValueError('Target ablation does not run source calibration.')
        campaign = read_json(session/'campaign_config.json')
        if campaign['checkpoint_sha256'] != checkpoint_hash or campaign['implementation_sha256'] != hashes():
            raise ValueError('Scheduled experiment changed.')
        experiment = SimpleNamespace(CANDIDATE=fixed.CANDIDATE, CONDITIONS=fixed.CONDITIONS, RULE=RULE,
            PARAMETER=fixed.PARAMETER, verify_calibration=verify_plan,
            dispersion_map=fixed.dispersion_map, gated_output=fixed.gated_output)
        target_worker(args, session, checkpoint, checkpoint_hash, experiment=experiment); return
    # Import after entry initialization to avoid a circular module import.
    from rma_fixed_fusion import command
    jobs = [(d, s) for d in args.datasets for s in args.shots]; audits = {}; reused = []
    for dataset, shot in jobs:
        audit, rows = history(args, fixed_history, dataset, shot, checkpoint_hash)
        audits.setdefault(dataset, {})[str(shot)] = audit; reused.extend(rows)
    session = new_directory(ROOT/'results/rma_fixed_fusion', 'all_fractions_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'testing', session=str(session))
    receipt = session/'evaluation_plan'; receipt.mkdir()
    plan = dict(rule=RULE, parameter_origin='predeclared_target_ablation', source_selected=False,
        fractions=[0., .125, .25, .375], rma_fraction=.125, seed=10, weight_epoch=14,
        checkpoint_sha256=checkpoint_hash, implementation_sha256=hashes(),
        reused_files_sha256={str(Path(r['Origin'])/'metrics_combined.csv'): digest(Path(r['Origin'])/'metrics_combined.csv') for r in reused})
    write_json(receipt/'locked_config.json', plan)
    write_json(session/'campaign_config.json', dict(plan, checkpoint=str(checkpoint), calibration=str(receipt),
        calibration_sha256=digest(receipt/'locked_config.json'), fixed_history=str(fixed_history),
        datasets=args.datasets, shots=args.shots, primary_comparator='G0_L1', source_selection_claimed=False))
    write_json(session/'input_audit.json', audits)
    write_csv(session/'reused_metrics.csv', list(reused[0]), reused)
    done = []; current = None
    try:
        merge(session, reused, done)
        for dataset, shot in jobs:
            current = [dataset, shot]
            write_json(session/'run_status.json', dict(status='testing_12_5_percent', complete=False, current=current,
                completed_jobs=done, reused_fractions=[0., .25, .375]))
            print(f'[12.5%补测 {len(done)+1}/{len(jobs)}] {dataset} {shot}-shot；0/25/37.5%已核对复用', flush=True)
            cmd = command(args, session, 'target', shot, dataset)
            cmd.extend(['--evaluate_all_fractions', '--fixed_history', str(fixed_history)])
            run_child(cmd)
            state = read_json(session/dataset/f'{shot}shot_seed10/run_status.json')
            if not all(state.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed')):
                raise ValueError('Incomplete round.')
            done.append((dataset, shot)); merge(session, reused, done)
            summarize(session, done, DATASETS, candidate=fixed.CANDIDATE); cleanup()
        write_json(session/'run_status.json', dict(status='completed', complete=True, completed_jobs=done,
            target_evaluation_complete=True, reused_fractions=[0., .25, .375], newly_evaluated_fraction=.125))
    except BaseException as error:
        failure_status(session, error)
        print(f'停止：固定12.5%，当前={current}，完成{len(done)}/{len(jobs)}；{session}', flush=True); raise
    print(f'Completed target fraction ablation: {session}', flush=True)
