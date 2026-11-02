"""Evaluate original A and joint-trained C on identical features and supports."""
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
from rma_joint_utils import (
    JOINT, THREE_ONLY, JointBuffer, clone_adapters, encode_learned_text,
    evaluate_candidate, load_joint_adapters, validate_joint_checkpoint,
)
from rma_utils import FIELDS, check_frozen, cleanup, digest, failure_status, freeze_base, run_child, save_tables
from test_rma import preflight


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', required=True, help='C joint-training best.pth, containing all four adapters.')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--disable_rma', action='store_true', help='Use UPDATED three adapters without RMA; this is not original A recovery.')
    p.add_argument('--save_maps', action='store_true', help='Save selected examples only, not full-test matrices.')
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
    internal = [args._session is not None, args._dataset is not None, args._shot is not None]
    if any(internal) and not all(internal):
        p.error('Supply all internal worker fields together.')
    for name in ('checkpoint', 'visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def save_example(args, source, output, sample, image, mask, raw, maps, index):
    import numpy as np
    relative = Path(sample['Category'])/sample['Sample Type']/sample['Sample Key']
    arrays = {n: v[index].detach().cpu().numpy() for n, v in maps.items()}
    raw_arrays = {n: v[index].detach().cpu().numpy() for n, v in raw.items()}
    checks = []
    for stage, values, key in (('raw', raw_arrays, 'Raw Maps'), ('evaluated', arrays, 'Evaluated Maps')):
        previous = np.load(Path(source['source'])/sample[key]/'FINAL.npy', allow_pickle=False)
        passed = previous.shape == values['baseline'].shape and np.isfinite(previous).all() and np.allclose(
            previous, values['baseline'], atol=1e-6, rtol=1e-5)
        checks.append(dict(Image=sample['Image Path'], Stage=stage, Passed=bool(passed)))
        if args.save_maps:
            directory = output/'raw_maps'/relative/stage
            directory.mkdir(parents=True)
            for name, value in values.items():
                np.save(directory/f'{name}.npy', value, allow_pickle=False)
    if args.save_visualizations:
        save_panels(output/'visualizations'/relative.with_suffix('.png'), image, mask.cpu().numpy(),
                    raw_arrays, arrays, list(maps), 2, 'Original A / joint-trained C')
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
    validate_joint_checkpoint(payload, digest(source['config']['checkpoint_path']),
                              digest(source['config']['pretrained_model']), source['config'])
    initialized = initialize(source, output, args.num_workers, logger)
    config, model, textual, visual, pq, original_text, memory, loader, dpam, device = initialized
    frozen = freeze_base(initialized)
    adapters = clone_adapters(initialized)
    load_joint_adapters(adapters, payload)
    rma = ReferenceMatchingAdapter(**payload['model_spec']).to(device).eval()
    rma.load_state_dict(payload['rma'], strict=True)
    rma.requires_grad_(False)
    del payload
    with torch.no_grad():
        candidate_text = encode_learned_text(model, adapters[0])
    candidate = THREE_ONLY if args.disable_rma else JOINT
    pixels, images, score_rows, checks, matrix_checks = [], [], [], [], []
    selected = {path_key(r['Image Path']): r for r in source['diagnostics']}
    visited, closed, buffer, count, batches = set(), set(), None, 0, 0

    def persist():
        for filename, rows in (('image_scores.csv', score_rows), ('baseline_score_checks.csv', checks),
                               ('baseline_map_checks.csv', matrix_checks)):
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
        for items in tqdm(loader, desc=f'{config.dataset} {config.k_shots}-shot / original A + {candidate}'):
            paths, categories = items['img_path'], items['cls_name']
            features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
            pf, pp = prompt_association(*memory, categories)
            old, old_maps, old_scores, maxima, native, fixed, _ = infer_batch(
                config, features, patches, pf, pp, textual, visual, pq, original_text, paths)
            versions = [v._version for v in [features, pf, *patches, *pp]]
            candidate_raw, candidate_map, candidate_score = evaluate_candidate(
                config, adapters, candidate_text, rma, features, patches, pf, pp, paths,
                without_rma=args.disable_rma)
            if versions != [v._version for v in [features, pf, *patches, *pp]]:
                raise RuntimeError('C modified shared frozen feature inputs.')
            raw = {'baseline': old['baseline'], candidate: candidate_raw}
            maps = {'baseline': old_maps['baseline'], candidate: candidate_map}
            scores = {'baseline': old_scores['baseline'], candidate: candidate_score}
            for name in maps:
                if maps[name].shape != items['img_mask'][:, 0].shape or scores[name].shape != (len(paths),):
                    raise ValueError('Unexpected candidate shape.')
                if not torch.isfinite(raw[name]).all() or not torch.isfinite(maps[name]).all() or not torch.isfinite(scores[name]).all():
                    raise ValueError('Nonfinite candidate output.')
            masks = (items['img_mask'][:, 0] > .5).to(torch.uint8)
            for i, category in enumerate(categories):
                expected = source['manifest'][count]
                label = int(items['anomaly'][i])
                if path_key(paths[i]) != path_key(expected['Image Path']) or category != expected['Category'] or label != int(expected['Sample Type']=='Anomaly'):
                    raise ValueError('Historical query/order/label mismatch.')
                if buffer is None or buffer.category != category:
                    if buffer is not None:
                        flush()
                        closed.add(buffer.category)
                    if category in closed:
                        raise ValueError('Noncontiguous category order.')
                    buffer = JointBuffer(category, candidate)
                buffer.append(masks[i], label, maps, scores, i)
                row = {'Category': category, 'Image Path': paths[i], 'Label': label,
                    **{f'{n} Global': float(v[i]) for n, v in native.items()},
                    'Fixed Global Mean': float(fixed[i]), 'baseline Map Max': float(maxima['baseline'][i]),
                    **{f'{n} Image Score': float(v[i]) for n, v in scores.items()}}
                previous = control['scores'][count]
                names = ['VA Global', 'TA Global', 'PQA Global', 'Fixed Global Mean', 'baseline Map Max', 'baseline Image Score']
                passed = all(math.isclose(row[n], float(previous[n]), abs_tol=1e-6, rel_tol=1e-5) for n in names)
                checks.append({'Image Path': paths[i], 'Passed': passed})
                score_rows.append(row)
                if not passed:
                    persist()
                    write_json(output/'failed_score_comparison.json', dict(current=row, historical=previous))
                    raise RuntimeError('Original A image scores differ from history.')
                key = path_key(paths[i])
                if key in selected:
                    matrix_checks.extend(save_example(args, source, output, selected[key], items['img'][i], masks[i], raw, maps, i))
                    visited.add(key)
                    if not all(r['Passed'] for r in matrix_checks):
                        persist()
                        raise RuntimeError('Original A selected matrices differ from history.')
                count += 1
            batches += 1
            del features, patches, pf, pp, old, old_maps, old_scores, raw, maps, scores
    if buffer is None or count != len(source['manifest']) or visited != set(selected):
        raise ValueError('Incomplete query/selected-example coverage.')
    flush()
    table = save_tables(output, pixels, images)
    metric_checks = compare_baseline(source, table, .001)
    write_csv(output/'baseline_reference_comparison.csv', list(metric_checks[0]), metric_checks)
    if not all(r['Passed'] for r in metric_checks):
        raise RuntimeError('Original A full-precision metrics differ from history.')
    check_frozen(frozen)
    if digest(args.checkpoint) != checkpoint_hash:
        raise ValueError('C checkpoint changed during evaluation.')
    return dict(status='completed', complete=True, training_scheme=JOINT, candidate=candidate,
        sample_count=count, query_encoder_calls=batches, baseline_parity_passed=True,
        frozen_integrity_passed=True, without_rma=args.disable_rma,
        note='Removing RMA from C does not restore original adapters; original A stays a separate output.')


def summarize(session, jobs, candidate):
    rows, deltas, retention = [], [], []
    for dataset, shot in jobs:
        table = read_csv(session/dataset/f'{shot}shot_seed10'/'metrics_combined.csv')
        rows.extend(dict(Dataset=dataset, Shot=shot, **r) for r in table)
        means = {r['Condition']: r for r in table if r['Category']=='Mean'}
        values = {f: float(means[candidate][f])-float(means['baseline'][f]) for f in FIELDS}
        deltas.append(dict(Dataset=dataset, Shot=shot, Condition=candidate, Comparator='baseline', **values, Net=sum(values.values())))
    for filename, keep_mean in (('metrics_summary.csv', True), ('metrics_by_category.csv', False)):
        write_csv(session/filename, ['Dataset', 'Shot', 'Category', 'Condition', *FIELDS],
                  [r for r in rows if (r['Category']=='Mean') == keep_mean])
    write_csv(session/'deltas_vs_original.csv', ['Dataset', 'Shot', 'Condition', 'Comparator', *FIELDS, 'Net'], deltas)
    for shot in sorted({s for _, s in jobs}):
        group = [r for r in deltas if r['Shot']==shot]
        positive, net = sum(r['Net']>0 for r in group), sum(r['Net'] for r in group)
        complete = {r['Dataset'] for r in group} == set(DATASETS)
        retention.append(dict(Shot=shot, PositiveDatasets=positive, CompletedDatasets=len(group),
                              Net=net, SixDatasetsComplete=complete, Retain=bool(complete and positive>=3 and net>0)))
    write_csv(session/'retention_by_shot.csv', list(retention[0]), retention)


def worker(args):
    session = project_path(args._session, ROOT)
    output = session/args._dataset/f'{args._shot}shot_seed10'
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('rma-joint-test')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'test.log', encoding='utf-8')
    logger.addHandler(handler)
    try:
        source, control, audit = prepare(args, args._dataset, args._shot)
        if audit != read_json(session/'input_audit.json')[args._dataset][str(args._shot)]:
            raise ValueError('Historical inputs changed after preflight.')
        checkpoint_hash = digest(args.checkpoint)
        if checkpoint_hash != read_json(session/'campaign_config.json')['checkpoint_sha256']:
            raise ValueError('C checkpoint changed after preflight.')
        write_json(output/'experiment_config.json', dict(**source['config'], training_scheme=JOINT,
            joint_checkpoint=str(args.checkpoint), joint_sha256=checkpoint_hash,
            disable_rma=args.disable_rma, save_candidate_maps=args.save_maps))
        write_json(output/'input_audit.json', audit)
        write_csv(output/'references.csv', ['Category', 'Image Path'], [{'Category': c, 'Image Path': p} for c, p in source['reference_paths']])
        write_csv(output/'sample_manifest.csv', list(source['manifest'][0]), source['manifest'])
        write_json(output/'run_status.json', dict(status='running', complete=False))
        status = inference(args, source, control, output, logger, checkpoint_hash)
        write_json(output/'run_status.json', dict(**status, dataset=IDS[args._dataset], k_shots=args._shot, seed=10))
    except BaseException as error:
        failure_status(output, error)
        logger.exception('C evaluation stopped.')
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
    jobs, audit = preflight(args)
    if args._output:
        session = project_path(args._output, ROOT)
        if session.parent != (ROOT/'results/test_rma_joint').resolve():
            raise ValueError('Internal output must be a new test_rma_joint session.')
        session.mkdir(parents=True, exist_ok=False)
    else:
        session = new_directory(ROOT/'results'/Path(__file__).stem, 'six_datasets_seed10')
    write_json(session/'input_audit.json', audit)
    candidate = THREE_ONLY if args.disable_rma else JOINT
    write_json(session/'campaign_config.json', dict(training_scheme=JOINT, candidate=candidate,
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint), datasets=args.datasets,
        shots=args.shots, seed=10, comparator='original A', disable_rma=args.disable_rma))
    done = []
    try:
        for dataset, shot in jobs:
            write_json(session/'run_status.json', dict(status='running', complete=False, completed_jobs=done, current=[dataset, shot]))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--checkpoint', str(args.checkpoint),
                       '--_session', str(session), '--_dataset', dataset, '--_shot', str(shot)]
            for name in ('history_session', 'visa_source_root', 'visa_control_root', 'num_workers'):
                command.extend([f'--{name}', str(getattr(args, name))])
            for name in ('disable_rma', 'save_maps'):
                if getattr(args, name):
                    command.append(f'--{name}')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            print(f'[{len(done)+1}/{len(jobs)}] {dataset} {shot}-shot / {candidate} -> {session}', flush=True)
            run_child(command)
            if read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json').get('complete') is not True:
                raise RuntimeError('Incomplete C evaluation round.')
            done.append((dataset, shot))
            summarize(session, done, candidate)
        write_json(session/'run_status.json', dict(status='completed', complete=True, training_scheme=JOINT, completed_jobs=done))
    except BaseException as error:
        failure_status(session, error)
        raise
    print(f'C evaluation completed: {session}', flush=True)


if __name__ == '__main__':
    main()
