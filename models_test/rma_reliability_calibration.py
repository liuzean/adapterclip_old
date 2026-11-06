"""MVTec source-only inference calibration; no optimizer or parameter updates."""
import logging
from pathlib import Path

from fusion_ablation_utils import path_key, read_csv, read_json, write_csv, write_json
from rma_output_ablation_utils import route_outputs
from rma_reliability_utils import (BETAS, RULE, SELECTION, ReliabilityBuffer, dispersion_map,
                                   gated_output, implementation_hashes, load_frozen, metric_checks,
                                   select_beta, source_inputs)
from rma_utils import check_frozen, cleanup, digest, failure_status, save_tables


def worker(args, session, checkpoint, checkpoint_hash, *, experiment=None):
    """One held-out MVTec shot in one process; all four strengths share features."""
    import torch
    from fusion_ablation import infer_batch
    from test import prompt_association
    from tqdm import tqdm
    rule = RULE if experiment is None else experiment.RULE
    choices = [(0., 'G0_L1'), *[(b, f'beta_{int(b)}') for b in BETAS[1:]]] if experiment is None else experiment.CHOICES
    diagnostic = dispersion_map if experiment is None else experiment.dispersion_map
    fuse = gated_output if experiment is None else experiment.gated_output
    output = session/'calibration'/f'{args._shot}shot_seed10'
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('rma-reliability-source'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'calibration.log', encoding='utf-8'); logger.addHandler(handler)
    try:
        source, split, audit = source_inputs(checkpoint)
        campaign = read_json(session/'campaign_config.json')
        if audit != campaign['source_audit'] or digest(checkpoint) != checkpoint_hash:
            raise ValueError('Source/checkpoint changed since preflight.')
        write_json(output/'source_audit.json', audit)
        write_json(output/'experiment_config.json', dict(source['config'], evaluation_shot=args._shot,
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, rule=rule, candidate_parameters=choices,
            **({'betas': list(BETAS)} if experiment is None else {'fractions': [v for v, _ in choices]}),
            num_workers=args.num_workers))
        root = Path(source['config']['test_data_path'])
        manifest = [dict(Category=r['cls_name'], Image=str((root/r['img_path']).resolve()), Label=int(r['anomaly']))
                    for r in split['validation_queries']]
        write_csv(output/'sample_manifest.csv', list(manifest[0]), manifest)
        write_json(output/'references.json', split['validation_supports'])
        initialized, adapters, rma, text, snapshots = load_frozen(source, output, checkpoint, args.num_workers, logger)
        config, model, _, _, _, _, memory, loader, dpam, device = initialized
        # Same 60 normal references encoded in batch2, same validation query order
        # and batch2 as new-D source_validation; ONLY its first S refs used below.
        loader.dataset.data_all = split['validation_queries']
        loader.dataset.length = len(manifest)
        names = (*[name for _, name in choices], 'G1_L1')
        pixels, images, scores_saved, count, buffer, closed = [], [], [], 0, None, set()

        def flush():
            p, i = buffer.finish(device); pixels.extend(p); images.extend(i)
            save_tables(output, pixels, images)
            write_csv(output/'image_scores.csv', list(scores_saved[0]), scores_saved)
            cleanup()

        write_json(output/'run_status.json', dict(status='calibrating', complete=False, shot=args._shot))
        with torch.no_grad():
            for items in tqdm(loader, desc=f'MVTec validation {args._shot}-shot / four strengths'):
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf = pf[:, :args._shot].to(device)
                pp = [p[:, :args._shot].to(device) for p in pp]
                old, maps0, scores0, _, _, fixed, extra = infer_batch(config, features, patches, pf, pp, *adapters, text, items['img_path'])
                g, l = rma(patches, pp)
                raw, maps, scores, _ = route_outputs(old, maps0, scores0, fixed, extra, g, l, config.sigma)
                dispersion = diagnostic(patches, pp, config.image_size, chunk=rma.spec['chunk'])
                endpoint = (raw['G0_L1'], maps['G0_L1'], scores['G0_L1'])
                for value, name in choices[1:]:
                    raw[name], maps[name], scores[name], _ = fuse(old, fixed, extra, l, dispersion, value, config.sigma, endpoint)
                for i, category in enumerate(items['cls_name']):
                    label = int(items['anomaly'][i]); expected = manifest[count]
                    if category != expected['Category'] or path_key(items['img_path'][i]) != path_key(expected['Image']) or label != expected['Label']:
                        raise ValueError('Source query identity/order/label changed.')
                    if buffer is None or category != buffer.category:
                        if buffer is not None:
                            flush(); closed.add(buffer.category)
                        if category in closed:
                            raise ValueError('Noncontiguous source classes.')
                        buffer = ReliabilityBuffer(category, conditions=names)
                    buffer.append(items['img_mask'][i, 0] > .5, label, maps, scores, i)
                    scores_saved.append(dict(Category=category, Image=items['img_path'][i], Label=label,
                                              **{n: float(scores[n][i]) for n in names}))
                    count += 1
                del features, patches, pf, pp, old, maps0, scores0, g, l, raw, maps, scores, dispersion, endpoint, extra
        if buffer is None or count != len(manifest):
            raise ValueError('Incomplete source validation coverage.')
        flush()
        table = save_tables(output, pixels, images)
        historical = checkpoint.parent/'validation'/'epoch_014'/f'{args._shot}shot'
        metric_checks(output, table, historical, [('G1_L1', 'scratch')])
        for snapshot in snapshots:
            check_frozen(snapshot)
        if digest(checkpoint) != checkpoint_hash or source_inputs(checkpoint)[2] != audit:
            raise ValueError('Source inputs/checkpoint changed during calibration.')
        write_json(output/'run_status.json', dict(status='completed', complete=True, shot=args._shot,
            sample_count=count, source_endpoint_parity_passed=True, frozen_integrity_passed=True))
    except BaseException as error:
        failure_status(output, error); logger.exception('Source calibration stopped.'); raise
    finally:
        cleanup(); handler.close(); logger.removeHandler(handler)


def lock(session, checkpoint, checkpoint_hash):
    """Select once on source. Target labels/metrics never enter this function."""
    directory = session/'calibration'
    tables = {}
    for shot in (1, 2, 4):
        child = directory/f'{shot}shot_seed10'
        state = read_json(child/'run_status.json')
        if state.get('complete') is not True or state.get('source_endpoint_parity_passed') is not True:
            raise ValueError('All three source calibration rounds must pass.')
        tables[shot] = read_csv(child/'metrics_combined.csv')
    beta, rows = select_beta(tables)
    write_csv(directory/'source_selection.csv', list(rows[0]), rows)
    write_json(directory/'locked_config.json', dict(rule=RULE, betas=list(BETAS), beta=beta, seed=10,
        weight_epoch=14, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        source_audit=source_inputs(checkpoint)[2], implementation_sha256=implementation_hashes(), selection=SELECTION,
        metric_sha256={str(s): digest(directory/f'{s}shot_seed10'/'metrics_combined.csv') for s in (1, 2, 4)},
        fallback_to_G0_L1=beta == 0, target_labels_used=False,
        note='Same source validation already selected epoch14; not an independent unbiased evaluation.'))
    write_json(directory/'run_status.json', dict(status='completed', complete=True, selected_beta=beta))
    print(f'MVTec源域校准完成：beta={beta:g}；0表示保持G0_L1、不降权。', flush=True)
    return directory
