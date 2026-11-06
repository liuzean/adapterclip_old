"""MVTec源域局部输出误差归因：原A、新D三分支、RMA单图、25%融合图。

固定新D第14轮、seed10和原345张验证查询/60张正常参考；默认1/2/4-shot。
只诊断，不训练、不选比例、不测试目标数据集；每shot独立子进程清理。
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, save_panels, write_csv, write_json
from new_d_without_rma import reference_checkpoint
from rma_local_diagnostics_utils import CONDITIONS, THRESHOLDS, measure, summaries
from rma_output_ablation_utils import route_outputs
from rma_reliability_utils import ReliabilityBuffer, load_frozen, metric_checks, source_inputs
from rma_utils import check_frozen, cleanup, digest, failure_status, record_pipeline_progress, run_child, save_tables


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--new_d_run', default='results/test_rma_scratch_new/six_datasets_seed10')
    p.add_argument('--source_history', default='results/rma_fixed_fusion/six_datasets_seed10/calibration')
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--vis_per_type', type=int, default=2, help='Fixed first N normal/anomaly samples per source category.')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save_maps', action='store_true', help='Selected source samples only, not full-test matrices.')
    p.add_argument('--_session', help=argparse.SUPPRESS)
    p.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.shots != sorted(set(args.shots)) or args.num_workers < 0 or args.vis_per_type < 0:
        p.error('Ascending unique shots and nonnegative workers/sample count required.')
    if bool(args._session) != (args._shot is not None):
        p.error('Internal session/shot required together.')
    for key in ('new_d_run', 'source_history'):
        setattr(args, key, project_path(getattr(args, key), ROOT))
    return args


def selection(split, per_type):
    counts = {}; chosen = set()
    for r in split['validation_queries']:
        key = r['cls_name'], int(r['anomaly'])
        counts[key] = counts.get(key, 0)+1
        if counts[key] <= per_type:
            chosen.add((r['cls_name'], r['img_path']))
    return chosen


def history_audit(args, checkpoint_hash, audit):
    lock = read_json(args.source_history/'locked_config.json')
    if (read_json(args.source_history/'run_status.json').get('complete') is not True
            or lock['checkpoint_sha256'] != checkpoint_hash or lock['source_audit'] != audit):
        raise ValueError('Historical source controls must have identical checkpoint and source inputs.')
    for shot in args.shots:
        directory = args.source_history/f'{shot}shot_seed10'
        if read_json(directory/'run_status.json').get('complete') is not True:
            raise ValueError('Missing source control round.')
        if digest(directory/'metrics_combined.csv') != lock['metric_sha256'][str(shot)]:
            raise ValueError('Historical source metrics changed after locking.')


def worker(args, checkpoint, checkpoint_hash):
    import numpy as np
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import smooth_maps
    from fusion_ablation import infer_batch
    from test import prompt_association
    from tqdm import tqdm
    session = project_path(args._session, ROOT)
    session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    output = session/f'{args._shot}shot_seed10'; output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_status.json', dict(status='initializing', complete=False, shot=args._shot, processed_queries=0))
    logger = logging.getLogger('rma-local-diagnostics'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'diagnostics.log', encoding='utf-8'); logger.addHandler(handler)
    try:
        source, split, audit = source_inputs(checkpoint)
        campaign = read_json(session/'campaign_config.json')
        if campaign['source_audit'] != audit or campaign['checkpoint_sha256'] != checkpoint_hash or campaign['code_sha256'] != code_hashes():
            raise ValueError('Scheduled inputs/weights/code changed.')
        history_audit(args, checkpoint_hash, audit)
        write_json(output/'experiment_config.json', dict(source['config'], evaluation_shot=args._shot, seed=10,
            new_d_checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, weight_epoch=14,
            conditions=CONDITIONS, thresholds=THRESHOLDS, rma_local_fraction=.25,
            result_dir=str(output), vis_per_type=args.vis_per_type, num_workers=args.num_workers,
            save_maps=args.save_maps, save_visualizations=args.save_visualizations))
        write_json(output/'source_audit.json', audit); write_json(output/'references.json', split['validation_supports'])
        effective = []
        for category in split['classes']:
            effective.extend([r for r in split['validation_supports'] if r['cls_name'] == category][:args._shot])
        write_json(output/'effective_references.json', effective)
        root = Path(source['config']['test_data_path'])
        manifest = [dict(Category=r['cls_name'], Image=str((root/r['img_path']).resolve()), Label=int(r['anomaly'])) for r in split['validation_queries']]
        write_csv(output/'sample_manifest.csv', list(manifest[0]), manifest)
        chosen = selection(split, args.vis_per_type)
        write_json(output/'selected_samples.json', [r for r in split['validation_queries'] if (r['cls_name'], r['img_path']) in chosen])
        initialized, adapters, rma, text, snapshots = load_frozen(source, output, checkpoint, args.num_workers, logger)
        config, model, a_t, a_v, a_pq, a_text, memory, loader, dpam, device = initialized
        loader.dataset.data_all = split['validation_queries']; loader.dataset.length = len(manifest)
        region_rows, gap_rows, effect_rows, score_rows = [], [], [], []
        pixels, images = [], []; buffer = None; closed = set(); visited = set(); count = batches = 0

        def persist():
            for file, rows in [('region_stats.csv', region_rows), ('foreground_background_gaps.csv', gap_rows),
                               ('paired_errors.csv', effect_rows), ('image_scores.csv', score_rows)]:
                if rows:
                    write_csv(output/file, list(rows[0]), rows)
            summaries(output, region_rows, effect_rows, gap_rows)

        def flush():
            p, i = buffer.finish(device); pixels.extend(p); images.extend(i)
            save_tables(output, pixels, images); persist(); cleanup()
            write_json(output/'run_status.json', dict(status='diagnosing', complete=False, shot=args._shot,
                processed_queries=count, last_completed_category=buffer.category))

        with torch.no_grad():
            for items in tqdm(loader, desc=f'MVTec local error attribution {args._shot}-shot'):
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf = pf[:, :args._shot].to(device); pp = [p[:, :args._shot].to(device) for p in pp]
                versions = [v._version for v in (features, pf, *patches, *pp)]
                a_raw, a_maps, a_scores, *_ = infer_batch(config, features, patches, pf, pp, a_t, a_v, a_pq, a_text, items['img_path'])
                old, maps0, scores0, _, _, fixed, extra = infer_batch(config, features, patches, pf, pp, *adapters, text, items['img_path'])
                g, l = rma(patches, pp)
                raw, maps, scores, _ = route_outputs(old, maps0, scores0, fixed, extra, g, l, config.sigma)
                raw['baseline'] = a_raw['baseline']; maps['baseline'] = a_maps['baseline']; scores['baseline'] = a_scores['baseline']
                raw['RMA_only'] = l.softmax(1)[:, 1]
                maps['RMA_only'] = smooth_maps(raw['RMA_only'], config.sigma).cpu()
                scores['RMA_only'] = fusion_fun([g.softmax(-1)[:, 1], maps['RMA_only'].flatten(1).max(1).values.to(device)], fusion_type='harmonic_mean').cpu()
                if versions != [v._version for v in (features, pf, *patches, *pp)]:
                    raise ValueError('Shared CLIP features mutated.')
                masks = (items['img_mask'][:, 0] > .5).cpu().numpy()
                for i, category in enumerate(items['cls_name']):
                    label = int(items['anomaly'][i]); expected = manifest[count]
                    if category != expected['Category'] or path_key(items['img_path'][i]) != path_key(expected['Image']) or label != expected['Label']:
                        raise ValueError('Source identity/order/label changed.')
                    if buffer is None or category != buffer.category:
                        if buffer is not None:
                            flush(); closed.add(buffer.category)
                        if category in closed:
                            raise ValueError('Noncontiguous source categories.')
                        buffer = ReliabilityBuffer(category, conditions=(*CONDITIONS, 'G1_L1'))
                    buffer.append(items['img_mask'][i, 0] > .5, label, maps, scores, i)
                    stages = {'raw': {n: raw[n][i].detach().cpu().numpy() for n in CONDITIONS},
                              'evaluated': {n: maps[n][i].detach().cpu().numpy() for n in CONDITIONS}}
                    r, gaps, effects = measure(items['img_path'][i], category, label, masks[i], stages, args._shot)
                    region_rows.extend(r); gap_rows.extend(gaps); effect_rows.extend(effects)
                    score_rows.append(dict(Category=category, Image=items['img_path'][i], Label=label, DefectPixels=int(masks[i].sum()),
                        **{f'{n} Image Score': float(scores[n][i]) for n in (*CONDITIONS, 'G1_L1')}))
                    sample = split['validation_queries'][count]; key = category, sample['img_path']
                    if key in chosen:
                        # Keep defect-type directories: different types may share 000.png.
                        relative = Path(sample['img_path']).with_suffix('')
                        if args.save_visualizations:
                            save_panels(output/'visualizations'/relative.with_suffix('.png'), items['img'][i], masks[i],
                                stages['raw'], stages['evaluated'], list(CONDITIONS), 3, 'MVTec source local error attribution')
                        if args.save_maps:
                            directory = output/'selected_maps'/relative; directory.mkdir(parents=True)
                            for stage, values in stages.items():
                                np.savez_compressed(directory/f'{stage}.npz', mask=masks[i], **values)
                        visited.add(key)
                    count += 1
                batches += 1
                del features, patches, pf, pp, old, maps0, scores0, raw, maps, scores, g, l, a_raw, a_maps, a_scores, extra
        if buffer is None or count != len(manifest) or visited != chosen:
            raise ValueError('Incomplete source coverage/selected samples.')
        flush(); table = save_tables(output, pixels, images)
        metric_checks(output, table, args.source_history/f'{args._shot}shot_seed10',
            [('G0_L0', 'fraction_0'), ('G0_L1', 'G0_L1'), ('G1_L1', 'G1_L1')])
        for snapshot in snapshots:
            check_frozen(snapshot)
        if digest(checkpoint) != checkpoint_hash or source_inputs(checkpoint)[2] != audit:
            raise ValueError('Source/weights changed during diagnosis.')
        write_json(output/'run_status.json', dict(status='completed', complete=True, shot=args._shot, queries=count,
            selected_samples=len(visited), encoder_calls=batches, source_endpoint_parity_passed=True, frozen_integrity_passed=True))
    except BaseException as error:
        failure_status(output, error); logger.exception('Local diagnostics stopped.'); raise
    finally:
        cleanup(); handler.close(); logger.removeHandler(handler)


def code_hashes():
    return {n: digest(ROOT/'models_test'/n) for n in ('rma_local_diagnostics.py', 'rma_local_diagnostics_utils.py',
        'rma_reliability_utils.py', 'rma_output_ablation_utils.py', 'fusion_ablation.py', 'rma_utils.py')}


def main(argv=None):
    args = parse_args(argv)
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    if args._session:
        worker(args, checkpoint, checkpoint_hash); return
    source, split, audit = source_inputs(checkpoint); history_audit(args, checkpoint_hash, audit)
    session = new_directory(ROOT/'results'/Path(__file__).stem, 'mvtec_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'diagnosing', session=str(session))
    write_json(session/'campaign_config.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        weight_epoch=14, seed=10, shots=args.shots, source_audit=audit, code_sha256=code_hashes(),
        conditions=CONDITIONS, thresholds=THRESHOLDS, rma_local_fraction=.25, source_history=str(args.source_history),
        num_workers=args.num_workers, vis_per_type=args.vis_per_type, save_maps=args.save_maps,
        save_visualizations=args.save_visualizations, source_config=source['config'],
        note='Original A trained on these source queries; this is error attribution, not unbiased model ranking.'))
    done = []; current = None
    try:
        for shot in args.shots:
            current = shot
            write_json(session/'run_status.json', dict(status='diagnosing', complete=False, current_shot=shot, completed_shots=done))
            print(f'[{len(done)+1}/{len(args.shots)}] MVTec {shot}-shot / 四图局部误差归因', flush=True)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--_session', str(session),
                       '--_shot', str(shot), '--shots', str(shot)]
            for key in ('new_d_run', 'source_history', 'num_workers', 'vis_per_type'):
                command.extend([f'--{key}', str(getattr(args, key))])
            if args.save_maps:
                command.append('--save_maps')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            run_child(command)
            state = read_json(session/f'{shot}shot_seed10/run_status.json')
            if not all(state.get(k) is True for k in ('complete', 'source_endpoint_parity_passed', 'frozen_integrity_passed')):
                raise ValueError('Diagnosis round did not pass all checks.')
            done.append(shot); cleanup()
        for filename in ('metrics_combined.csv', 'region_summary.csv', 'paired_error_summary.csv', 'gap_summary.csv'):
            combined = [dict(Shot=s, **r) for s in done for r in read_csv(session/f'{s}shot_seed10'/filename)]
            if combined:
                write_csv(session/filename, list(combined[0]), combined)
        write_json(session/'run_status.json', dict(status='completed', complete=True, completed_shots=done))
    except BaseException as error:
        failure_status(session, error)
        print(f'停止于MVTec {current}-shot，已完成={done}；记录：{session}', flush=True); raise
    print(f'Completed source diagnostics: {session}', flush=True)


if __name__ == '__main__':
    main()
