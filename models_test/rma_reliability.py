"""新D epoch14：只给RMA局部图增加正常参考匹配可靠性约束，不训练。

先用已有MVTec验证集1/2/4-shot确定一个统一强度，再锁定它评价六数据集。
直接对照G0_L1，同时报告原A和完整新D。默认seed10、dataset先于shot。
"""
import argparse
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS, prepare
from fusion_ablation_utils import (compare_baseline, new_directory, path_key, project_path,
                                   read_csv, read_json, save_panels, write_csv, write_json)
from new_d_without_rma import reference_checkpoint
from rma_output_ablation_utils import audit_history, route_outputs
from rma_reliability_utils import (BETAS, CANDIDATE, CONDITIONS, RULE, SELECTION, ReliabilityBuffer,
    dispersion_map, gated_output, implementation_hashes, load_frozen, metric_checks,
    source_inputs, summarize, verify_calibration)
from rma_utils import check_frozen, cleanup, digest, failure_status, run_child, save_tables


def parse_args(argv=None, *, description=__doc__):
    p = argparse.ArgumentParser(description=description, allow_abbrev=False)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--new_d_run', default='results/test_rma_scratch_new/six_datasets_seed10')
    p.add_argument('--off_run', default='results/new_d_without_rma/six_datasets_seed10')
    p.add_argument('--ablation_run', default='results/rma_output_ablation/six_datasets_seed10')
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--calibration', help='Reuse a completed, unchanged source calibration directory (not a checkpoint).')
    p.add_argument('--calibrate_only', action='store_true', help='Run three source validation rounds and lock beta, without targets.')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_maps', action='store_true', help='Selected sample matrices only; no full-test matrix cache.')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--_session', help=argparse.SUPPRESS)
    p.add_argument('--_phase', choices=('calibrate', 'target'), help=argparse.SUPPRESS)
    p.add_argument('--_dataset', choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_workers < 0 or args.shots != sorted(set(args.shots)) or len(args.datasets) != len(set(args.datasets)):
        p.error('Unique datasets, ascending unique shots, nonnegative workers required.')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        p.error('history_session must be one directory name.')
    if args.calibration and args.calibrate_only:
        p.error('--calibration and --calibrate_only cannot be combined.')
    if any(x is not None for x in (args._session, args._phase, args._dataset, args._shot)):
        if not args._session or not args._phase or args._shot is None or (args._phase == 'target') != (args._dataset is not None):
            p.error('Incomplete internal worker fields.')
    for name in ('new_d_run', 'off_run', 'ablation_run', 'visa_source_root', 'visa_control_root', 'calibration'):
        if getattr(args, name) is not None:
            setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def target_inputs(args, dataset, shot, checkpoint_hash):
    """Require historical A/full/off/G0_L1 to share exact inputs and weights."""
    source, control, audit = prepare(args, dataset, shot)
    relative = Path(dataset)/f'{shot}shot_seed10'
    audit_history(source, audit, args.new_d_run/relative, args.off_run/relative)
    campaign = read_json(args.ablation_run/'campaign_config.json')
    directory = args.ablation_run/relative
    if (read_json(args.ablation_run/'run_status.json').get('complete') is not True
            or campaign['checkpoint_sha256'] != checkpoint_hash or campaign['weight_epoch'] != 14
            or read_json(directory/'run_status.json').get('complete') is not True
            or read_json(directory/'input_audit.json') != audit
            or read_csv(directory/'sample_manifest.csv') != source['manifest']
            or [(r['Category'], r['Image Path']) for r in read_csv(directory/'references.csv')] != list(source['reference_paths'])):
        raise ValueError('G0_L1 history incomplete or input/weight identity differs.')
    config = read_json(directory/'experiment_config.json')
    keys = ('dataset', 'seed', 'k_shots', 'test_data_path', 'pretrained_model', 'checkpoint_path',
            'features_list', 'image_size', 'batch_size', 'sigma', 'n_ctx', 'vl_reduction',
            'pq_mid_dim', 'pq_context', 'visual_learner', 'textual_learner', 'pq_learner', 'fusion_type')
    if config['checkpoint_sha256'] != checkpoint_hash or any(config[k] != source['config'][k] for k in keys):
        raise ValueError('G0_L1 historical evaluation protocol changed.')
    return source, control, audit


def save_sample(args, source, output, sample, image, mask, raw, maps, dispersion, alpha, index, *, conditions=CONDITIONS, alphas=None):
    """Preserve baseline matrix checks; optional selected reliability matrices."""
    from test_rma_joint import save_example
    options = SimpleNamespace(**dict(vars(args), save_visualizations=False))
    checks = save_example(options, source, output, sample, image, mask, raw, maps, index)
    relative = Path(sample['Category'])/sample['Sample Type']/sample['Sample Key']
    if args.save_maps:
        import numpy as np
        directory = output/'reliability_maps'/relative.parent; directory.mkdir(parents=True, exist_ok=True)
        arrays = dict(rma_fraction=alpha[index].cpu().numpy())
        if dispersion is not None:
            arrays['dispersion'] = dispersion[index].cpu().numpy()
        if alphas is not None:
            arrays.update({f'rma_fraction_{n}': v[index].cpu().numpy() for n, v in alphas.items()})
        np.savez_compressed(directory/f'{relative.name}.npz', **arrays)
    if args.save_visualizations:
        arrays = {n: v[index].detach().cpu().numpy() for n, v in maps.items()}
        original = {n: v[index].detach().cpu().numpy() for n, v in raw.items()}
        title = 'Original A / G0_L1 / full new D / reliability gate' if conditions == CONDITIONS else 'Original A / G0_L1 / full new D / fixed local fraction'
        if any(n.startswith('beta_') for n in conditions):
            title = 'Original A / beta0 G0_L1 / full new D / beta4,16,64'
        if 'dual_output' in conditions:
            title = 'Original A / G0_L1 / full new D / local candidate / dual output'
        save_panels(output/'visualizations'/relative.with_suffix('.png'), image, mask.cpu().numpy(),
                    original, arrays, list(conditions), 2, title)
    return checks


def target_worker(args, session, checkpoint, checkpoint_hash, *, experiment=None):
    import torch
    from fusion_ablation import infer_batch
    from test import prompt_association
    from tqdm import tqdm
    candidate = CANDIDATE if experiment is None else experiment.CANDIDATE
    conditions = CONDITIONS if experiment is None else experiment.CONDITIONS
    rule = RULE if experiment is None else experiment.RULE
    parameter = 'beta' if experiment is None else experiment.PARAMETER
    verify = verify_calibration if experiment is None else experiment.verify_calibration
    diagnostic = dispersion_map if experiment is None else experiment.dispersion_map
    fuse = gated_output if experiment is None else experiment.gated_output
    parameter_grid = None if experiment is None else getattr(experiment, 'CANDIDATE_PARAMETERS', None)
    output_adapter = None if experiment is None else getattr(experiment, 'OUTPUT_ADAPTER', None)
    buffer_factory = ReliabilityBuffer if experiment is None else getattr(experiment, 'BUFFER_FACTORY', ReliabilityBuffer)
    final_check = None if experiment is None else getattr(experiment, 'FINAL_CHECK', None)
    output = session/args._dataset/f'{args._shot}shot_seed10'; output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('rma-reliability'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'test.log', encoding='utf-8'); logger.addHandler(handler)
    try:
        campaign = read_json(session/'campaign_config.json')
        calibration = Path(campaign['calibration'])
        locked = verify(calibration, checkpoint, checkpoint_hash)
        if digest(calibration/'locked_config.json') != campaign['calibration_sha256']:
            raise ValueError('Locked source selection changed after target preflight.')
        beta = locked[parameter]
        candidate_parameters = {candidate: beta} if parameter_grid is None else dict(parameter_grid)
        if candidate_parameters.get(candidate) != beta or not set(candidate_parameters).issubset(conditions):
            raise ValueError('Invalid candidate parameter grid.')
        source, control, audit = target_inputs(args, args._dataset, args._shot, checkpoint_hash)
        if audit != read_json(session/'input_audit.json')[args._dataset][str(args._shot)]:
            raise ValueError('Target inputs changed since preflight.')
        write_json(output/'experiment_config.json', dict(source['config'], checkpoint_sha256=checkpoint_hash,
            new_d_checkpoint=str(checkpoint), weight_epoch=14, rule=rule, **{parameter: beta}, conditions=list(conditions),
            calibration=str(calibration), calibration_sha256=campaign['calibration_sha256'], align_used=True,
            rma_global_used_by_candidate=False, candidate_parameters=candidate_parameters,
            max_rma_local_fraction=.25 if parameter == 'beta' else max(candidate_parameters.values()), num_workers=args.num_workers,
            save_maps=args.save_maps, save_visualizations=args.save_visualizations))
        if experiment is not None and hasattr(experiment, 'OUTPUT_DEFINITION'):
            saved_config = read_json(output/'experiment_config.json')
            saved_config['output_definition'] = experiment.OUTPUT_DEFINITION
            write_json(output/'experiment_config.json', saved_config)
        write_json(output/'input_audit.json', audit)
        write_csv(output/'references.csv', ['Category', 'Image Path'], [{'Category': c, 'Image Path': p} for c, p in source['reference_paths']])
        write_csv(output/'sample_manifest.csv', list(source['manifest'][0]), source['manifest'])
        write_json(output/'run_status.json', dict(status='initializing', complete=False))
        initialized, adapters, rma, candidate_text, snapshots = load_frozen(source, output, checkpoint, args.num_workers, logger)
        config, model, textual, visual, pq, text, memory, loader, dpam, device = initialized
        history_dir = args.ablation_run/args._dataset/f'{args._shot}shot_seed10'
        historical = read_csv(history_dir/'image_scores.csv')
        if len(historical) != len(source['manifest']):
            raise ValueError('Historical image-score coverage differs.')
        selected = {path_key(r['Image Path']): r for r in source['diagnostics']}
        scores_saved, checks, matrix_checks, pixels, images = [], [], [], [], []
        count = batches = 0
        buffer = None; closed = set(); visited = set()

        def persist():
            for filename, rows in [('image_scores.csv', scores_saved), ('endpoint_score_checks.csv', checks), ('baseline_map_checks.csv', matrix_checks)]:
                if rows:
                    write_csv(output/filename, list(rows[0]), rows)

        def flush():
            p, i = buffer.finish(device); pixels.extend(p); images.extend(i)
            save_tables(output, pixels, images); persist(); cleanup()

        with torch.no_grad():
            for items in tqdm(loader, desc=f'{args._dataset} {args._shot}-shot / RMA {parameter}={beta:g}'):
                paths = items['img_path']; classes = items['cls_name']
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, classes)
                a_raw, a_maps, a_scores, _, a_native, a_fixed, _ = infer_batch(config, features, patches, pf, pp, textual, visual, pq, text, paths)
                versions = [v._version for v in (features, pf, *patches, *pp)]
                old, maps0, scores0, _, _, fixed, extra = infer_batch(config, features, patches, pf, pp, *adapters, candidate_text, paths)
                g, l = rma(patches, pp)
                raw, maps, scores, _ = route_outputs(old, maps0, scores0, fixed, extra, g, l, config.sigma)
                dispersion = diagnostic(patches, pp, config.image_size, chunk=rma.spec['chunk'])
                alphas = {}
                endpoint = (raw['G0_L1'], maps['G0_L1'], scores['G0_L1'])
                for name, value in candidate_parameters.items():
                    raw[name], maps[name], scores[name], alphas[name] = fuse(old, fixed, extra, l,
                        dispersion, value, config.sigma, endpoint)
                alpha = alphas[candidate]
                raw['baseline'] = a_raw['baseline']; maps['baseline'] = a_maps['baseline']; scores['baseline'] = a_scores['baseline']
                if output_adapter is not None:
                    output_adapter(raw, maps, scores)
                raw = {n: raw[n] for n in conditions}; maps = {n: maps[n] for n in conditions}; scores = {n: scores[n] for n in conditions}
                if versions != [v._version for v in (features, pf, *patches, *pp)]:
                    raise ValueError('Candidate modified shared features.')
                for name in conditions:
                    if maps[name].shape != items['img_mask'][:, 0].shape or scores[name].shape != (len(paths),):
                        raise ValueError('Wrong output shape.')
                    if not all(torch.isfinite(v).all() for v in (raw[name], maps[name], scores[name])):
                        raise ValueError('Nonfinite output.')
                masks = (items['img_mask'][:, 0] > .5).to(torch.uint8)
                for i, category in enumerate(classes):
                    expected = source['manifest'][count]; label = int(items['anomaly'][i]); h = historical[count]
                    if (category != expected['Category'] or path_key(paths[i]) != path_key(expected['Image Path'])
                            or label != int(expected['Sample Type'] == 'Anomaly')
                            or category != h['Category'] or path_key(paths[i]) != path_key(h['Image Path']) or label != int(h['Label'])):
                        raise ValueError('Query identity/order/label changed.')
                    if buffer is None or category != buffer.category:
                        if buffer is not None:
                            flush(); closed.add(buffer.category)
                        if category in closed:
                            raise ValueError('Noncontiguous target classes.')
                        buffer = buffer_factory(category, conditions=conditions)
                    buffer.append(masks[i], label, maps, scores, i)
                    scores_saved.append(dict(Category=category, **{'Image Path': paths[i]}, Label=label,
                        **{f'{n} Image Score': float(scores[n][i]) for n in conditions},
                        DispersionMean=None if dispersion is None else float(dispersion[i].mean()), RMAFractionMean=float(alpha[i].mean()),
                        RMAFractionMin=float(alpha[i].min()), RMAFractionMax=float(alpha[i].max()),
                        **{'Beta' if parameter == 'beta' else 'RMAFraction': beta},
                        **({f'{n} RMAFraction{label}': float(getattr(v[i], method)()) for n, v in alphas.items()
                            for label, method in [('Mean', 'mean'), ('Min', 'min'), ('Max', 'max')]} if parameter_grid is not None else {})))
                    for name in ('baseline', 'G0_L1', 'G1_L1'):
                        passed = math.isclose(float(scores[name][i]), float(h[f'{name} Image Score']), abs_tol=1e-6, rel_tol=1e-5)
                        checks.append(dict(Image=paths[i], Condition=name, Passed=passed))
                    previous = control['scores'][count]
                    native_ok = all(math.isclose(float(a_native[n][i]), float(previous[f'{n} Global']), abs_tol=1e-6, rel_tol=1e-5) for n in ('VA', 'TA', 'PQA'))
                    native_ok = native_ok and math.isclose(float(a_fixed[i]), float(previous['Fixed Global Mean']), abs_tol=1e-6, rel_tol=1e-5)
                    checks.append(dict(Image=paths[i], Condition='A native globals', Passed=native_ok))
                    if not all(r['Passed'] for r in checks[-4:]):
                        persist(); raise ValueError('Original A/G0_L1/full new-D historical score parity failed.')
                    key = path_key(paths[i])
                    if key in selected:
                        matrix_checks.extend(save_sample(args, source, output, selected[key], items['img'][i], masks[i], raw, maps, dispersion, alpha, i,
                            conditions=conditions, alphas=alphas if parameter_grid is not None else None))
                        visited.add(key)
                        if not all(r['Passed'] for r in matrix_checks):
                            persist(); raise ValueError('Original-A selected map parity failed.')
                    count += 1
                batches += 1
                del features, patches, pf, pp, a_raw, a_maps, a_scores, old, maps0, scores0, g, l, raw, maps, scores, dispersion, alpha, alphas, endpoint, extra
        if buffer is None or count != len(source['manifest']) or visited != set(selected):
            raise ValueError('Incomplete target coverage.')
        flush(); table = save_tables(output, pixels, images)
        baseline = compare_baseline(source, table, .001)
        write_csv(output/'baseline_reference_comparison.csv', list(baseline[0]), baseline)
        if not all(r['Passed'] for r in baseline):
            raise ValueError('Original-A metric parity failed.')
        metric_checks(output, table, history_dir, [(n, n) for n in ('baseline', 'G0_L1', 'G1_L1')])
        for snapshot in snapshots:
            check_frozen(snapshot)
        if digest(checkpoint) != checkpoint_hash or digest(calibration/'locked_config.json') != campaign['calibration_sha256']:
            raise ValueError('Checkpoint/calibration changed during evaluation.')
        additional_checks = {} if final_check is None else final_check(output, table)
        write_json(output/'run_status.json', dict(status='completed', complete=True, sample_count=count,
            query_encoder_calls=batches, new_D_adapter_calls=batches, rma_calls=batches, **{parameter: beta},
            candidate_parameters=candidate_parameters,
            endpoint_parity_passed=True, frozen_integrity_passed=True, weight_epoch=14, align_used=True, **additional_checks))
    except BaseException as error:
        failure_status(output, error); logger.exception('RMA reliability evaluation stopped.'); raise
    finally:
        cleanup(); handler.close(); logger.removeHandler(handler)


def child_command(args, session, phase, shot, dataset=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--_session', str(session), '--_phase', phase, '--_shot', str(shot)]
    if dataset is not None:
        command.extend(['--_dataset', dataset])
    for key in ('new_d_run', 'off_run', 'ablation_run', 'history_session', 'visa_source_root', 'visa_control_root', 'num_workers'):
        command.extend([f'--{key}', str(getattr(args, key))])
    if args.save_maps:
        command.append('--save_maps')
    if not args.save_visualizations:
        command.append('--no-save_visualizations')
    return command


def main(argv=None):
    args = parse_args(argv)
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    if args._session:
        session = project_path(args._session, ROOT)
        session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
        campaign = read_json(session/'campaign_config.json')
        if checkpoint_hash != campaign['checkpoint_sha256'] or implementation_hashes() != campaign['implementation_sha256']:
            raise ValueError('Checkpoint/implementation changed since scheduling.')
        if args._phase == 'calibrate':
            from rma_reliability_calibration import worker
            worker(args, session, checkpoint, checkpoint_hash)
        else:
            target_worker(args, session, checkpoint, checkpoint_hash)
        return
    _, _, source_audit = source_inputs(checkpoint)
    session = new_directory(ROOT/'results'/Path(__file__).stem, 'six_datasets_seed10')
    campaign = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, weight_epoch=14, seed=10,
        rule=RULE, betas=list(BETAS), selection=SELECTION, source_audit=source_audit,
        implementation_sha256=implementation_hashes(), datasets=args.datasets, shots=args.shots,
        conditions=list(CONDITIONS), primary_comparator='G0_L1', target_labels_used_for_selection=False,
        num_workers=args.num_workers, save_maps=args.save_maps, save_visualizations=args.save_visualizations,
        history={n: str(getattr(args, n)) for n in ('new_d_run', 'off_run', 'ablation_run',
            'history_session', 'visa_source_root', 'visa_control_root')})
    write_json(session/'campaign_config.json', campaign)
    done = []; source_done = []; phase = 'calibration'; current = None
    try:
        if args.calibration:
            calibration = args.calibration
            verify_calibration(calibration, checkpoint, checkpoint_hash)
        else:
            (session/'calibration').mkdir()
            for shot in (1, 2, 4):
                current = ['MVTec_validation', shot]
                write_json(session/'run_status.json', dict(status='calibrating', complete=False, phase=phase,
                    current=current, completed_source_shots=source_done, completed_jobs=done))
                print(f'[源域校准 {len(source_done)+1}/3] MVTec {shot}-shot（不训练）', flush=True)
                run_child(child_command(args, session, 'calibrate', shot))
                source_done.append(shot); cleanup()
            from rma_reliability_calibration import lock
            calibration = lock(session, checkpoint, checkpoint_hash)
        locked = verify_calibration(calibration, checkpoint, checkpoint_hash)
        campaign.update(calibration=str(calibration), calibration_sha256=digest(calibration/'locked_config.json'), beta=locked['beta'])
        write_json(session/'campaign_config.json', campaign)
        if args.calibrate_only:
            write_json(session/'run_status.json', dict(status='calibration_only_completed', complete=True, target_evaluation_complete=False,
                calibration=str(calibration), beta=locked['beta']))
            print(f'仅完成源域校准，未评价目标集：{calibration}', flush=True); return
        # Target metadata and historical metrics are first accessed AFTER selection is locked.
        phase = 'target_preflight'; current = None
        jobs = [(d, s) for d in args.datasets for s in args.shots]; audits = {}
        for dataset, shot in jobs:
            _, _, audit = target_inputs(args, dataset, shot, checkpoint_hash)
            audits.setdefault(dataset, {})[str(shot)] = audit
        write_json(session/'input_audit.json', audits)
        phase = 'target'
        for dataset, shot in jobs:
            current = [dataset, shot]
            write_json(session/'run_status.json', dict(status='testing', complete=False, phase=phase, current=current,
                completed_source_shots=source_done, completed_jobs=done, calibration=str(calibration)))
            print(f'[目标评价 {len(done)+1}/{len(jobs)}] {dataset} {shot}-shot', flush=True)
            run_child(child_command(args, session, 'target', shot, dataset))
            state = read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json')
            if not all(state.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed')):
                raise ValueError('Target round did not pass all checks.')
            done.append((dataset, shot)); summarize(session, done, args.datasets); cleanup()
        write_json(session/'run_status.json', dict(status='completed', complete=True, target_evaluation_complete=True,
            completed_jobs=done, calibration=str(calibration), beta=locked['beta']))
    except BaseException as error:
        path = session/'run_status.json'
        state = read_json(path) if path.exists() else {}
        state.update(phase=phase, current=current, completed_source_shots=source_done, completed_jobs=done)
        write_json(path, state); failure_status(session, error)
        print(f'已停止：阶段={phase}，当前={current}，源域已完成={source_done}，目标已完成={len(done)}；记录：{session}', flush=True)
        raise
    print(f'Completed: {session}', flush=True)


if __name__ == '__main__':
    main()
