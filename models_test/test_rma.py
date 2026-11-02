"""Frozen original Baseline versus trained RMA, six datasets x 1/2/4 shots."""
import argparse
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS, IDS, prepare
from fusion_ablation_utils import (
    compare_baseline, new_directory, path_key, project_path, read_csv, read_json,
    save_panels, write_csv, write_json,
)
from rma_utils import (
    CONDITIONS, FIELDS, MetricsBuffer, check_frozen, cleanup, digest, failure_status,
    freeze_base, fuse_rma, run_child, save_tables, validate_checkpoint_payload,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Explicit trained best.pth; never guesses the latest run.')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--disable_rma', action='store_true', help='Candidate aliases the original three-branch baseline.')
    p.add_argument('--save_maps', action='store_true', help='Only selected normal/anomaly sample maps, not all samples.')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--_output', help=argparse.SUPPRESS)
    p.add_argument('--_session', help=argparse.SUPPRESS)
    p.add_argument('--_dataset', choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_workers < 0 or len(set(args.datasets)) != len(args.datasets) or args.shots != sorted(set(args.shots)):
        p.error('Nonnegative workers, unique datasets and ascending unique shots required.')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        p.error('history_session must be one directory name.')
    worker = [args._session is not None, args._dataset is not None, args._shot is not None]
    if any(worker) and not all(worker):
        p.error('All internal worker fields must be supplied together.')
    for name in ('checkpoint', 'visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def preflight(args):
    """Read-only protocol/reference audit before launching any GPU subprocess."""
    jobs = [(d, s) for d in args.datasets for s in args.shots]
    audits = {}
    for dataset, shot in jobs:
        _, _, audit = prepare(args, dataset, shot)
        audits.setdefault(dataset, {})[str(shot)] = audit
    return jobs, audits


def save_example(args, source, output, sample, image, mask, raw, maps, index):
    import numpy as np
    relative = Path(sample['Category'])/sample['Sample Type']/sample['Sample Key']
    arrays = {n: v[index].detach().cpu().numpy() for n, v in maps.items()}
    raw_arrays = {n: v[index].detach().cpu().numpy() for n, v in raw.items()}
    checks = []
    for stage, values, key in (('raw', raw_arrays, 'Raw Maps'), ('evaluated', arrays, 'Evaluated Maps')):
        previous = np.load(Path(source['source'])/sample[key]/'FINAL.npy', allow_pickle=False)
        current = values['baseline']
        passed = previous.shape == current.shape and np.isfinite(previous).all() and np.allclose(previous, current, atol=1e-6, rtol=1e-5)
        checks.append(dict(Image=sample['Image Path'], Stage=stage, Passed=bool(passed)))
        if args.save_maps:
            directory = output/'raw_maps'/relative/stage
            directory.mkdir(parents=True)
            for name, value in values.items():
                np.save(directory/f'{name}.npy', value, allow_pickle=False)
    if args.save_visualizations:
        save_panels(output/'visualizations'/relative.with_suffix('.png'), image, mask.cpu().numpy(),
                    raw_arrays, arrays, list(CONDITIONS), 2, 'Original Baseline / frozen base + RMA')
    return checks


def inference(args, source, control, output, logger, checkpoint_hash):
    import torch
    from fusion_ablation import infer_batch, initialize
    from rma_model import ReferenceMatchingAdapter
    from test import prompt_association
    from tools import setup_seed
    from tqdm import tqdm
    setup_seed(10)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_payload(payload, digest(source['config']['checkpoint_path']),
                                digest(source['config']['pretrained_model']), source['config'])
    # Preserve original baseline initialization RNG order before adding RMA.
    initialized = initialize(source, output, args.num_workers, logger)
    config, model, textual, visual, pq, text, memory, loader, dpam, device = initialized
    frozen = freeze_base(initialized)
    adapter = ReferenceMatchingAdapter(**payload['model_spec']).to(device).eval()
    adapter.load_state_dict(payload['rma'], strict=True)
    del payload
    pixels, images, scores_table, checks, matrix_checks = [], [], [], [], []
    selected = {path_key(r['Image Path']): r for r in source['diagnostics']}
    visited, closed, buffer, count, batches = set(), set(), None, 0, 0

    def persist():
        for filename, rows in (('baseline_score_checks.csv', checks), ('baseline_map_checks.csv', matrix_checks),
                               ('image_scores.csv', scores_table)):
            if rows:
                write_csv(output/filename, list(rows[0]), rows)

    def flush():
        p, i = buffer.finish(device)
        pixels.extend(p)
        images.extend(i)
        save_tables(output, pixels, images)
        persist()
        cleanup()

    with torch.no_grad():
        for items in tqdm(loader, desc=f'{config.dataset} {config.k_shots}-shot / Baseline + RMA'):
            paths, categories = items['img_path'], items['cls_name']
            features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
            pf, pp = prompt_association(*memory, categories)
            old, old_maps, old_scores, maxima, native, fixed, extra = infer_batch(
                config, features, patches, pf, pp, textual, visual, pq, text, paths)
            versions = [v._version for v in [*patches, *pp]]
            if args.disable_rma:
                candidate_raw, candidate_map, candidate_score = old['baseline'], old_maps['baseline'], old_scores['baseline']
            else:
                g, local = adapter(patches, pp)
                candidate_raw, candidate_map, candidate_score = fuse_rma(old, extra, fixed, g, local, config.sigma)
            if versions != [v._version for v in [*patches, *pp]]:
                raise RuntimeError('RMA modified shared feature inputs.')
            raw = dict(baseline=old['baseline'], rma=candidate_raw)
            maps = dict(baseline=old_maps['baseline'], rma=candidate_map)
            scores = dict(baseline=old_scores['baseline'], rma=candidate_score)
            if args.disable_rma and not (torch.equal(maps['baseline'], maps['rma']) and torch.equal(scores['baseline'], scores['rma'])):
                raise RuntimeError('Disabled recovery failed.')
            masks = (items['img_mask'][:, 0] > .5).to(torch.uint8)
            for index, category in enumerate(categories):
                expected = source['manifest'][count]
                label = int(items['anomaly'][index])
                if path_key(paths[index]) != path_key(expected['Image Path']) or category != expected['Category'] or label != int(expected['Sample Type']=='Anomaly'):
                    raise ValueError('Historical query identity/order/label mismatch.')
                if buffer is None or buffer.category != category:
                    if buffer is not None:
                        flush()
                        closed.add(buffer.category)
                    if category in closed:
                        raise ValueError('Noncontiguous query categories.')
                    buffer = MetricsBuffer(category)
                buffer.append(masks[index], label, maps, scores, index)
                row = {'Category': category, 'Image Path': paths[index], 'Label': label,
                       **{f'{n} Global': float(v[index]) for n, v in native.items()},
                       'Fixed Global Mean': float(fixed[index]), 'baseline Map Max': float(maxima['baseline'][index]),
                       **{f'{n} Image Score': float(v[index]) for n, v in scores.items()}}
                previous = control['scores'][count]
                names = ['VA Global', 'TA Global', 'PQA Global', 'Fixed Global Mean', 'baseline Map Max', 'baseline Image Score']
                passed = all(math.isclose(row[n], float(previous[n]), abs_tol=1e-6, rel_tol=1e-5) for n in names)
                checks.append({'Image Path': paths[index], 'Passed': passed})
                scores_table.append(row)
                if not passed:
                    persist()
                    write_json(output/'failed_score_comparison.json', dict(current=row, historical=previous))
                    raise RuntimeError('Original Baseline per-image scores differ from history.')
                key = path_key(paths[index])
                if key in selected:
                    matrix_checks.extend(save_example(args, source, output, selected[key], items['img'][index], masks[index], raw, maps, index))
                    visited.add(key)
                    if not all(c['Passed'] for c in matrix_checks):
                        persist()
                        raise RuntimeError('Original Baseline fixed-sample matrix differs from history.')
                count += 1
            batches += 1
            del features, patches, pf, pp, old, old_maps, old_scores, raw, maps, scores, extra
    if buffer is None or count != len(source['manifest']) or visited != set(selected):
        raise ValueError('Incomplete query/selected-sample coverage.')
    flush()
    combined = save_tables(output, pixels, images)
    metric_checks = compare_baseline(source, combined, .001)
    write_csv(output/'baseline_reference_comparison.csv', list(metric_checks[0]), metric_checks)
    write_csv(output/'baseline_score_checks.csv', list(checks[0]), checks)
    write_csv(output/'baseline_map_checks.csv', list(matrix_checks[0]), matrix_checks)
    write_csv(output/'image_scores.csv', list(scores_table[0]), scores_table)
    if not all(c['Passed'] for c in metric_checks):
        raise RuntimeError('Original Baseline full-precision metrics differ from history.')
    check_frozen(frozen)
    if digest(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('RMA checkpoint changed during evaluation.')
    return dict(status='completed', complete=True, sample_count=count, query_encoder_calls=batches,
                baseline_parity_passed=True, frozen_integrity_passed=True,
                disable_rma=args.disable_rma, disabled_recovery=True if args.disable_rma else None)


def summarize(session, jobs):
    rows, deltas = [], []
    for dataset, shot in jobs:
        table = read_csv(session/dataset/f'{shot}shot_seed10'/'metrics_combined.csv')
        rows.extend(dict(Dataset=dataset, Shot=shot, **r) for r in table)
        means = {r['Condition']: r for r in table if r['Category'] == 'Mean'}
        values = {f: float(means['rma'][f])-float(means['baseline'][f]) for f in FIELDS}
        deltas.append(dict(Dataset=dataset, Shot=shot, **values, Net=sum(values.values())))
    write_csv(session/'metrics_summary.csv', ['Dataset', 'Shot', 'Category', 'Condition', *FIELDS], [r for r in rows if r['Category']=='Mean'])
    write_csv(session/'metrics_by_category.csv', ['Dataset', 'Shot', 'Category', 'Condition', *FIELDS], [r for r in rows if r['Category']!='Mean'])
    write_csv(session/'deltas_vs_original.csv', ['Dataset', 'Shot', *FIELDS, 'Net'], deltas)
    retention = []
    for shot in sorted({s for _, s in jobs}):
        group = [r for r in deltas if r['Shot']==shot]
        positive = sum(r['Net'] > 0 for r in group)
        net = sum(r['Net'] for r in group)
        complete = {r['Dataset'] for r in group} == set(DATASETS)
        retention.append(dict(Shot=shot, CompletedDatasets=len(group), PositiveDatasets=positive,
                              Net=net, SixDatasetsComplete=complete, Retain=bool(complete and positive >= 3 and net > 0)))
    write_csv(session/'retention_by_shot.csv', list(retention[0]), retention)


def worker(args):
    session = project_path(args._session, ROOT)
    output = session/args._dataset/f'{args._shot}shot_seed10'
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('rma-test')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'test.log', encoding='utf-8')
    logger.addHandler(handler)
    try:
        source, control, audit = prepare(args, args._dataset, args._shot)
        campaign = read_json(session/'input_audit.json')
        if campaign[args._dataset][str(args._shot)] != audit:
            raise ValueError('Historical inputs changed after preflight.')
        checkpoint_hash = digest(args.checkpoint)
        if checkpoint_hash != read_json(session/'campaign_config.json')['checkpoint_sha256']:
            raise ValueError('RMA checkpoint changed after preflight.')
        write_json(output/'experiment_config.json', dict(**source['config'], rma_checkpoint=str(args.checkpoint),
                   rma_sha256=checkpoint_hash, disable_rma=args.disable_rma, save_candidate_maps=args.save_maps))
        write_json(output/'input_audit.json', audit)
        write_csv(output/'references.csv', ['Category', 'Image Path'], [{'Category': c, 'Image Path': p} for c, p in source['reference_paths']])
        write_csv(output/'sample_manifest.csv', list(source['manifest'][0]), source['manifest'])
        write_json(output/'run_status.json', dict(status='running', complete=False))
        status = inference(args, source, control, output, logger, checkpoint_hash)
        status.update(dataset=IDS[args._dataset], k_shots=args._shot, seed=10)
        write_json(output/'run_status.json', status)
    except BaseException as error:
        failure_status(output, error)
        logger.exception('Evaluation stopped.')
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
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    jobs, audits = preflight(args)
    if args._output:
        session = project_path(args._output, ROOT)
        if session.parent != (ROOT/'results/test_rma').resolve():
            raise ValueError('Internal output must be a new direct test_rma session.')
        session.mkdir(parents=True, exist_ok=False)
    else:
        session = new_directory(ROOT/'results'/Path(__file__).stem, 'six_datasets_seed10')
    write_json(session/'input_audit.json', audits)
    write_json(session/'campaign_config.json', dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        datasets=args.datasets, shots=args.shots, seed=10, comparator='original AdaptCLIP', disable_rma=args.disable_rma))
    done = []
    try:
        for dataset, shot in jobs:
            write_json(session/'run_status.json', dict(status='running', complete=False, completed_jobs=done, current=[dataset, shot]))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--checkpoint', str(args.checkpoint),
                '--_session', str(session), '--_dataset', dataset, '--_shot', str(shot)]
            for key in ('history_session', 'visa_source_root', 'visa_control_root', 'num_workers'):
                command.extend([f'--{key}', str(getattr(args, key))])
            for key in ('disable_rma', 'save_maps'):
                if getattr(args, key):
                    command.append(f'--{key}')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            print(f'[{len(done)+1}/{len(jobs)}] {dataset} {shot}-shot -> {session}', flush=True)
            run_child(command)
            status = read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json')
            if status.get('complete') is not True:
                raise RuntimeError('Incomplete evaluation round.')
            done.append((dataset, shot))
            summarize(session, done)
        write_json(session/'run_status.json', dict(status='completed', complete=True, completed_jobs=done))
    except BaseException as error:
        failure_status(session, error)
        raise
    print(f'Evaluation completed: {session}', flush=True)


if __name__ == '__main__':
    main()
