"""实验H：SH=(1-lambda)*SE+lambda*G0；像素定位完整固定为E。

冻结新D第14轮，不训练。MVTec源域1/2/4-shot选择一个统一lambda；
六目标集1/2/4-shot只做已保存分数的离线评价，不重新编码、不新存像素矩阵。
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import MODEL_KEYS, new_directory, path_key, project_path, read_csv, read_json, write_json as plain_json
from new_d_without_rma import reference_checkpoint
from rma_reliability import target_inputs
from rma_reliability_utils import source_inputs
from rma_score_coordination_utils import IMAGE_FIELDS, LAMBDAS, SELECTION, image_metrics, mix_scores, save, source_selection
from rma_target_diagnostics_utils import score_index
from rma_utils import FIELDS, cleanup, digest, run_child


def write_json(path, data):
    path = Path(path); temp = path.with_name(path.name+'.tmp')
    plain_json(temp, data); temp.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, default in dict(new_d_run='results/test_rma_scratch_new/six_datasets_seed10',
                              off_run='results/new_d_without_rma/six_datasets_seed10',
                              ablation_run='results/rma_output_ablation/six_datasets_seed10',
                              e_run='results/rma_fixed_dual_output/six_datasets_seed10',
                              source_history='results/rma_local_diagnostics/mvtec_seed10',
                              visa_source_root='results/branch_diagnostics',
                              visa_control_root='results/dual_output/visa_seed10').items():
        parser.add_argument('--'+name, default=default)
    parser.add_argument('--history_session', default='multi_dataset_seed10')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume_from')
    parser.add_argument('--status')
    parser.add_argument('--_session', help=argparse.SUPPRESS)
    parser.add_argument('--_output', help=argparse.SUPPRESS)
    parser.add_argument('--_phase', choices=('source', 'target'), help=argparse.SUPPRESS)
    parser.add_argument('--_dataset', choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.datasets != sorted(set(args.datasets), key=DATASETS.index) or args.shots != sorted(set(args.shots))
            or args.num_workers < 0 or Path(args.history_session).name != args.history_session or args.history_session in ('.', '..')):
        parser.error('Ordered unique datasets/shots and nonnegative workers required.')
    fields = (args._session, args._output, args._phase, args._shot)
    if any(value is not None for value in fields) and (not all(value is not None for value in fields)
            or (args._phase == 'target') != (args._dataset is not None)):
        parser.error('Incomplete internal worker fields.')
    if args.status and args.resume_from:
        parser.error('Use --status or --resume_from separately.')
    tokens = sys.argv[1:] if argv is None else argv
    if args.resume_from and any(token.startswith('--') and token.split('=', 1)[0] != '--resume_from' for token in tokens):
        parser.error('Resume restores recorded options; pass --resume_from alone.')
    for name in ('new_d_run', 'off_run', 'ablation_run', 'e_run', 'source_history', 'visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def code_hashes():
    from rma_reliability_utils import implementation_hashes
    result = implementation_hashes()
    for name in ('rma_score_coordination.py', 'rma_score_coordination_utils.py', 'rma_target_diagnostics_utils.py',
                 'rma_local_diagnostics_utils.py', 'context_alignment.py', 'new_d_without_rma.py', 'rma_scratch_common.py'):
        result[name] = digest(ROOT/'models_test'/name)
    return result


def target_history(args, dataset, shot, weight_hash):
    source, _, audit = target_inputs(args, dataset, shot, weight_hash)
    rel = Path(dataset)/f'{shot}shot_seed10'; folder = args.e_run/rel
    if (not read_json(folder/'run_status.json').get('complete') or read_json(folder/'input_audit.json') != audit
            or read_csv(folder/'sample_manifest.csv') != source['manifest']
            or [(row['Category'], row['Image Path']) for row in read_csv(folder/'references.csv')] != source['reference_paths']):
        raise ValueError('E input/coverage identity mismatch.')
    config = read_json(folder/'experiment_config.json')
    if (config.get('checkpoint_sha256') != weight_hash or config.get('rma_fraction') != .125
            or any(config[key] != source['config'][key] for key in (*MODEL_KEYS, 'k_shots'))):
        raise ValueError('E evaluation protocol/weight mismatch.')
    e_scores = read_csv(folder/'image_scores.csv'); original = read_csv(args.ablation_run/rel/'image_scores.csv')
    e_index, index = score_index(e_scores), score_index(original)
    identities = {(row['Category'], path_key(row['Image Path'])) for row in source['manifest']}
    if set(e_index) != identities or set(index) != identities:
        raise ValueError('Historical score query coverage differs.')
    result = []
    for row in source['manifest']:
        key = row['Category'], path_key(row['Image Path']); a, e = index[key], e_index[key]
        label = int(row['Sample Type'] == 'Anomaly')
        if (int(a['Label']) != label or int(e['Label']) != label
                or float(a['G0_L1 Image Score']) != float(e['dual_output Image Score'])):
            raise ValueError('E image donor/label differs from recorded G0_L1.')
        result.append(dict(Category=row['Category'], **{'Image Path': row['Image Path']}, Label=label,
                           E=float(e['dual_output Image Score']), G0=float(a['G0'])))
    return source, audit, result, read_csv(folder/'metrics_combined.csv')


def preflight(args):
    checkpoint, weight_hash = reference_checkpoint(args)
    _, _, source_audit = source_inputs(checkpoint)
    e = read_json(args.e_run/'campaign_config.json')
    if (not read_json(args.e_run/'run_status.json').get('complete') or e.get('mode') != 'fixed125'
            or e.get('checkpoint_sha256') != weight_hash or e.get('weight_epoch') != 14 or e.get('rma_fraction') != .125):
        raise ValueError('Requires the completed exact experiment E.')
    files = {str(checkpoint): weight_hash}
    for name, value in e['implementation_sha256'].items():
        path = ROOT/'models_test'/name
        if digest(path) != value:
            raise ValueError(f'Historical E implementation changed: {name}')
        files[str(path)] = value
    source_campaign = read_json(args.source_history/'campaign_config.json')
    if (not read_json(args.source_history/'run_status.json').get('complete')
            or source_campaign['source_audit'] != source_audit or source_campaign['checkpoint_sha256'] != weight_hash):
        raise ValueError('Source historical G0_L1 endpoint must match checkpoint and validation split.')
    for shot in (1, 2, 4):
        folder = args.source_history/f'{shot}shot_seed10'
        if not read_json(folder/'run_status.json').get('complete'):
            raise ValueError('Incomplete source endpoint history.')
        for name in ('image_scores.csv', 'sample_manifest.csv', 'effective_references.json', 'source_audit.json', 'run_status.json'):
            files[str(folder/name)] = digest(folder/name)
    for dataset in args.datasets:
        for shot in args.shots:
            source, audit, _, _ = target_history(args, dataset, shot, weight_hash)
            for run in (args.e_run, args.ablation_run):
                folder = run/dataset/f'{shot}shot_seed10'
                for name in ('image_scores.csv', 'metrics_combined.csv', 'input_audit.json', 'experiment_config.json',
                             'references.csv', 'sample_manifest.csv', 'run_status.json'):
                    files[str(folder/name)] = digest(folder/name)
            for kind in ('source', 'control'):
                for name, value in audit[kind+'_hashes'].items():
                    files[str(Path(audit[kind])/name)] = value
            files[str(Path(source['config']['test_data_path'])/'meta.json')] = audit['metadata_sha256']
    for name in ('model_manifest.json', 'run_status.json', 'source_split.json'):
        files[str(checkpoint.parent/name)] = digest(checkpoint.parent/name)
    manifest = read_json(checkpoint.parent/'model_manifest.json')
    for key, expected in (('checkpoint_path', 'base_sha256'), ('pretrained_model', 'pretrained_sha256')):
        files[manifest['base_protocol'][key]] = manifest[expected]
    files[str(Path(manifest['base_protocol']['test_data_path'])/'meta.json')] = manifest['source_meta_sha256']
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=weight_hash, weight_epoch=14, seed=10,
                source_audit=source_audit, historical_sha256=files, implementation_sha256=code_hashes(),
                lambda_grid=LAMBDAS, selection=SELECTION, training=False,
                pixel_source='Exact saved experiment E 12.5% final map metrics; not recomputed',
                target_mode='Offline recorded scores, not fresh inference')


def verify(lock):
    if lock['implementation_sha256'] != code_hashes():
        raise ValueError('Implementation changed after scheduling.')
    for name, value in lock['historical_sha256'].items():
        if digest(name) != value:
            raise ValueError(f'Historical/source/weight file changed: {name}')


def source_worker(args, session, output, lock):
    import torch
    from fusion_ablation import infer_batch
    from rma_output_ablation_utils import route_outputs
    from rma_reliability_utils import load_frozen
    from rma_utils import check_frozen
    from test import prompt_association
    from tqdm import tqdm
    source, split, audit = source_inputs(Path(lock['checkpoint']))
    if audit != lock['source_audit']:
        raise ValueError('Source validation split changed.')
    folder = args.source_history/f'{args._shot}shot_seed10'
    historical = read_csv(folder/'image_scores.csv')
    index = {(row['Category'], path_key(row['Image'])): row for row in historical}
    effective = [row for category in split['classes'] for row in
                 [row for row in split['validation_supports'] if row['cls_name'] == category][:args._shot]]
    if read_json(folder/'effective_references.json') != effective:
        raise ValueError('Source effective reference identity/order mismatch.')
    write_json(output/'references.json', effective)
    write_json(output/'source_audit.json', audit)
    logger = logging.getLogger('H-source'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'source.log', encoding='utf-8'); logger.addHandler(handler)
    try:
        initialized, adapters, rma, text, snapshots = load_frozen(source, output, Path(lock['checkpoint']), args.num_workers, logger)
        config, model, _, _, _, _, memory, loader, dpam, device = initialized
        loader.dataset.data_all = split['validation_queries']; loader.dataset.length = len(split['validation_queries'])
        root = Path(source['config']['test_data_path']); rows = []; checks = []
        with torch.no_grad():
            for items in tqdm(loader, desc=f'H source {args._shot}-shot'):
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf, pp = pf[:, :args._shot].to(device), [p[:, :args._shot].to(device) for p in pp]
                shared = (features, pf, *patches, *pp); versions = [v._version for v in shared]
                old, maps0, scores0, _, _, native, extra = infer_batch(config, features, patches, pf, pp, *adapters, text, items['img_path'])
                global_logits, local_logits = rma(patches, pp)
                _, _, scores, _ = route_outputs(old, maps0, scores0, native, extra, global_logits, local_logits, config.sigma)
                for i, category in enumerate(items['cls_name']):
                    expected = split['validation_queries'][len(rows)]; image = items['img_path'][i]; label = int(items['anomaly'][i])
                    if category != expected['cls_name'] or path_key(image) != path_key(root/expected['img_path']) or label != int(expected['anomaly']):
                        raise ValueError('Source query order/identity/label mismatch.')
                    old_row = index[category, path_key(image)]
                    delta = float(scores['G0_L1'][i])-float(old_row['G0_L1 Image Score'])
                    checks.append(dict(Category=category, Image=image, Delta=delta, Passed=abs(delta) <= 1e-6))
                    if int(old_row['Label']) != label or abs(delta) > 1e-6:
                        raise ValueError('Source E-image endpoint changed.')
                    rows.append(dict(Category=category, **{'Image Path': image}, Label=label,
                                     E=float(scores['G0_L1'][i]), G0=float(native[i])))
                write_json(output/'run_status.json', dict(complete=False, phase='source', processed_queries=len(rows)))
                if versions != [v._version for v in shared]:
                    raise ValueError('Shared CLIP features mutated.')
                del features, patches, pf, pp, shared, old, maps0, scores0, scores, native, extra, global_logits, local_logits
        if len(rows) != len(split['validation_queries']) or len(index) != len(rows):
            raise ValueError('Incomplete source query coverage.')
        save(output/'image_scores.csv', rows); save(output/'endpoint_checks.csv', checks)
        table = []
        for coefficient in LAMBDAS:
            values = mix_scores(torch.tensor([row['E'] for row in rows]), torch.tensor([row['G0'] for row in rows]), coefficient)
            scored = [dict(row, H=float(value)) for row, value in zip(rows, values)]
            table.extend(dict(row, Lambda=coefficient) for row in image_metrics(scored, 'H', f'lambda_{coefficient:g}'))
        save(output/'source_image_metrics.csv', table)
        for snapshot in snapshots:
            check_frozen(snapshot)
        if source_inputs(Path(lock['checkpoint']))[2] != audit:
            raise ValueError('Source changed during calibration.')
        return ['image_scores.csv', 'endpoint_checks.csv', 'source_image_metrics.csv', 'source_audit.json', 'references.json']
    finally:
        handler.close(); logger.removeHandler(handler)


def lock_selection(session, lock):
    tables = {shot: read_csv(Path(read_json(session/'run_status.json')['job_results'][f'source/{shot}'])/'source_image_metrics.csv') for shot in (1, 2, 4)}
    coefficient, rows = source_selection(tables)
    save(session/'source_selection.csv', rows)
    state = read_json(session/'run_status.json')
    proof = {str(Path(state['job_results'][f'source/{shot}'])/'source_image_metrics.csv'):
             digest(Path(state['job_results'][f'source/{shot}'])/'source_image_metrics.csv') for shot in (1, 2, 4)}
    value = dict(coefficient=coefficient, lambda_grid=LAMBDAS, selection=SELECTION,
                 checkpoint_sha256=lock['checkpoint_sha256'], source_audit=lock['source_audit'],
                 implementation_sha256=lock['implementation_sha256'], source_metrics_sha256=proof,
                 target_labels_used=False, note='Source split also selected epoch14; not independent blind validation.')
    path = session/'locked_config.json'
    if path.exists() and read_json(path) != json_compatible(value):
        raise ValueError('Existing source lock differs; never reselect on targets.')
    write_json(path, value)
    return value


def json_compatible(value):
    import json
    return json.loads(json.dumps(value))


def target_worker(args, session, output, lock):
    import torch
    selected = read_json(session/'locked_config.json')
    if selected != json_compatible(lock_selection(session, lock)):
        raise ValueError('Coefficient differs from the source-only selection.')
    if (selected['checkpoint_sha256'] != lock['checkpoint_sha256'] or selected['implementation_sha256'] != lock['implementation_sha256']
            or selected['source_audit'] != lock['source_audit'] or selected['target_labels_used'] is not False):
        raise ValueError('Invalid source-only lock.')
    for name, value in selected['source_metrics_sha256'].items():
        if digest(name) != value:
            raise ValueError('Source metrics changed after selection.')
    coefficient = selected['coefficient']
    source, audit, rows, historical = target_history(args, args._dataset, args._shot, lock['checkpoint_sha256'])
    values = mix_scores(torch.tensor([row['E'] for row in rows]), torch.tensor([row['G0'] for row in rows]), coefficient)
    scored = [dict(row, H=float(value)) for row, value in zip(rows, values)]
    checks = []; recomputed_e = image_metrics(rows, 'E', 'E')
    lookup = {(row['Category'], row['Condition']): row for row in historical}
    for row in recomputed_e:
        for field in IMAGE_FIELDS:
            delta = row[field]-float(lookup[row['Category'], 'dual_output'][field])
            checks.append(dict(Category=row['Category'], Metric=field, DeltaPP=delta, Passed=abs(delta) <= .001))
    if not all(row['Passed'] for row in checks):
        raise ValueError('Offline original image evaluator does not recover E.')
    h_images = image_metrics(scored, 'H', 'H')
    table = []
    for row in h_images:
        category = row['Category']; e = lookup[category, 'dual_output']
        for condition, donor in (('A', 'baseline'), ('E', 'dual_output'), ('G0_L1', 'G0_L1'), ('new_D', 'G1_L1')):
            table.append(dict(Category=category, Condition=condition,
                              **{field: float(lookup[category, donor][field]) for field in FIELDS}))
        # At lambda0 preserve all E metric values exactly, including numerical rounding.
        image = {field: float(e[field]) if coefficient == 0 else row[field] for field in IMAGE_FIELDS}
        table.append(dict(Category=category, Condition='H', **image, **{field: float(e[field]) for field in FIELDS[3:]}))
    save(output/'metrics_combined.csv', table); save(output/'image_scores.csv', scored)
    save(output/'E_metric_checks.csv', checks)
    write_json(output/'input_audit.json', audit)
    save(output/'sample_manifest.csv', source['manifest'])
    save(output/'references.csv', [dict(Category=c, **{'Image Path': p}) for c, p in source['reference_paths']])
    write_json(output/'experiment_config.json', dict(source['config'], coefficient=coefficient, checkpoint=lock['checkpoint'],
        weight_epoch=14, target_mode='offline_recorded_scores', pixel_source=str(args.e_run/args._dataset/f'{args._shot}shot_seed10'),
        full_pixel_metrics_recomputed=False, pixel_map_changed=False, lambda_source='MVTec locked_config.json'))
    return ['metrics_combined.csv', 'image_scores.csv', 'E_metric_checks.csv', 'input_audit.json', 'sample_manifest.csv', 'references.csv', 'experiment_config.json']


def worker(args):
    session, output = project_path(args._session, ROOT), project_path(args._output, ROOT)
    session.relative_to((ROOT/'results'/Path(__file__).stem).resolve()); output.relative_to(session)
    lock = read_json(session/'campaign_config.json'); verify(lock)
    if any(output.iterdir()):
        raise ValueError('Need a fresh empty attempt directory.')
    write_json(output/'run_status.json', dict(complete=False, phase=args._phase, dataset=args._dataset, shot=args._shot))
    try:
        names = source_worker(args, session, output, lock) if args._phase == 'source' else target_worker(args, session, output, lock)
        verify(lock)
        write_json(output/'run_status.json', dict(complete=True, phase=args._phase, dataset=args._dataset, shot=args._shot,
            checkpoint_sha256=lock['checkpoint_sha256'], code=lock['implementation_sha256'], artifacts={name: digest(output/name) for name in names}))
    except BaseException as error:
        old = read_json(output/'run_status.json'); write_json(output/'run_status.json', dict(old, complete=False, error=str(error)))
        raise
    finally:
        cleanup()


def summarize(args, session, state):
    metrics, deltas, decisions = [], [], []
    for dataset in args.datasets:
        for shot in args.shots:
            rows = read_csv(Path(state['job_results'][f'target/{dataset}/{shot}'])/'metrics_combined.csv')
            metrics.extend(dict(Dataset=dataset, Shot=shot, **row) for row in rows)
            means = {row['Condition']: row for row in rows if row['Category'] == 'Mean'}
            for comparator in ('E', 'A', 'G0_L1', 'new_D'):
                change = {field: float(means['H'][field])-float(means[comparator][field]) for field in FIELDS}
                deltas.append(dict(Dataset=dataset, Shot=shot, Comparator=comparator, **change, Net=sum(change.values())))
    for comparator in ('E', 'A', 'G0_L1', 'new_D'):
        for shot in args.shots:
            group = [row for row in deltas if row['Shot'] == shot and row['Comparator'] == comparator]
            positive = sum(row['Net'] > 0 for row in group); net = sum(row['Net'] for row in group)
            decisions.append(dict(Comparator=comparator, Shot=shot, ImprovedDatasets=positive, Net=net,
                                  Pass=positive >= 3 and net > 0 if len(group) == 6 else None))
    save(session/'metrics_summary.csv', metrics); save(session/'dataset_deltas.csv', deltas); save(session/'retention_by_shot.csv', decisions)
    eligible = len(args.datasets) == 6 and args.shots == [1, 2, 4]
    primary = [row for row in decisions if row['Comparator'] == 'E']
    write_json(session/'retention_overall.json', dict(primary='E', eligible=eligible,
        passing_shots=sum(row['Pass'] is True for row in primary), retained=sum(row['Pass'] is True for row in primary) >= 2 if eligible else None,
        pixel_metrics_reused=True, source_selected_coefficient=read_json(session/'locked_config.json')['coefficient']))


def options(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            if not key.startswith('_') and key not in ('resume_from', 'status')}


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        path = project_path(args.status, ROOT); state = read_json(path/'run_status.json')
        print(path, state)
        if state.get('current_output') and (Path(state['current_output'])/'run_status.json').exists():
            print(read_json(Path(state['current_output'])/'run_status.json'))
        return
    if args._session:
        worker(args); return
    if args.resume_from:
        session = project_path(args.resume_from, ROOT); session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
        lock = read_json(session/'campaign_config.json')
        for key, value in lock['options'].items():
            setattr(args, key, project_path(value, ROOT) if key.endswith('_run') or key.endswith('_root') or key == 'source_history' else value)
        verify(lock); state = read_json(session/'run_status.json')
    else:
        print('H preflight: frozen epoch14, source validation, recorded target E/G0 identities.', flush=True)
        lock = preflight(args); lock['options'] = options(args)
        session = new_directory(ROOT/'results'/Path(__file__).stem, 'experiment_H_seed10')
        write_json(session/'campaign_config.json', lock)
        state = dict(complete=False, phase='source', completed_jobs=[], job_results={}, current=None, current_output=None)
        write_json(session/'run_status.json', state)
    jobs = [('source', None, shot) for shot in (1, 2, 4)]+[('target', d, s) for d in args.datasets for s in args.shots]
    try:
        for phase, dataset, shot in jobs:
            if phase == 'target':
                selected = lock_selection(session, lock)
                print(f"Locked source lambda={selected['coefficient']:g}; target labels do not select it.", flush=True)
            key = f'source/{shot}' if phase == 'source' else f'target/{dataset}/{shot}'
            prior = state['job_results'].get(key)
            if prior:
                try:
                    previous = read_json(Path(prior)/'run_status.json')
                except (OSError, ValueError):
                    previous = {}
                if previous.get('complete'):
                    if previous.get('code') != lock['implementation_sha256'] or previous.get('checkpoint_sha256') != lock['checkpoint_sha256']:
                        raise ValueError('Completed worker code/weight identity mismatch.')
                    for name, value in previous['artifacts'].items():
                        if digest(Path(prior)/name) != value:
                            raise ValueError('Completed worker output changed.')
                    if key not in state['completed_jobs']:
                        state['completed_jobs'].append(key)
                    print('Reuse verified complete:', key); continue
            parent = session/'source' if phase == 'source' else session/'targets'/dataset
            output = new_directory(parent, f'{shot}shot_seed10')
            state.update(phase=phase, current=key, current_output=str(output), complete=False)
            state['job_results'][key] = str(output); write_json(session/'run_status.json', state)
            command = [sys.executable, str(Path(__file__).resolve()), '--_session', str(session), '--_output', str(output),
                       '--_phase', phase, '--_shot', str(shot)]
            if dataset:
                command.extend(['--_dataset', dataset])
            for name, value in lock['options'].items():
                command.extend(['--'+name, *map(str, value)] if isinstance(value, list) else ['--'+name, str(value)])
            write_json(session/'last_command.json', dict(command=command, job=key))
            print(f'[{jobs.index((phase, dataset, shot))+1}/{len(jobs)}] {key}; independent process.', flush=True)
            run_child(command)
            if not read_json(output/'run_status.json').get('complete'):
                raise ValueError('Child exited without completing checks.')
            cleanup(); state['completed_jobs'].append(key); write_json(session/'run_status.json', state)
        verify(lock); summarize(args, session, state)
        state.update(complete=True, phase='completed', current=None, current_output=None); state.pop('error', None)
        write_json(session/'run_status.json', state)
        print('Experiment H completed:', session)
    except BaseException as error:
        state.update(complete=False, error=f'{type(error).__name__}: {error}'); write_json(session/'run_status.json', state)
        print(f"Stopped at {state['current']}; completed={state['completed_jobs']}\nSession={session}\nWorker={state['current_output']}", flush=True)
        raise


if __name__ == '__main__':
    main()
