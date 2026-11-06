"""实验G：SDD定位与part_glass图像评分的目标域误差归因。

先复用六数据集已有六指标，再补两数据集1/2/4-shot的冻结诊断推理。
原A、新流程三适配器、新D关闭RMA、G0_L1、E；新D始终第14轮。
不训练、不选阈值、不改融合。按图归约统计，不缓存全测试集像素矩阵。
"""
import argparse
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import MODEL_KEYS, new_directory, path_key, project_path, read_csv, read_json, write_csv
from fusion_ablation_utils import write_json as write_plain_json
from new_d_without_rma import reference_checkpoint
from rma_reliability import target_inputs
from rma_target_diagnostics_utils import CONDITIONS, PAIRS, harmonic_score, observe, ranking_reports, save_rows, score_index, selected_keys
from rma_utils import FIELDS, cleanup, digest, run_child

TARGETS = ('SDD', 'part_glass')
BASE_OPTIONS = ('new_d_run', 'off_run', 'ablation_run', 'three_run', 'e_run',
                'history_session', 'visa_source_root', 'visa_control_root',
                'datasets', 'shots', 'num_workers', 'vis_per_type', 'save_visualizations', 'save_maps')


def write_json(path, value):
    """Replace this experiment's metadata atomically, preserving prior progress."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    write_plain_json(temporary, value)
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--datasets', nargs='+', choices=TARGETS, default=list(TARGETS))
    parser.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--new_d_run', default='results/test_rma_scratch_new/six_datasets_seed10')
    parser.add_argument('--off_run', default='results/new_d_without_rma/six_datasets_seed10')
    parser.add_argument('--ablation_run', default='results/rma_output_ablation/six_datasets_seed10')
    parser.add_argument('--three_run', default='results/test_new_d_three/six_datasets_seed10')
    parser.add_argument('--e_run', default='results/rma_fixed_dual_output/six_datasets_seed10')
    parser.add_argument('--history_session', default='multi_dataset_seed10')
    parser.add_argument('--visa_source_root', default='results/branch_diagnostics')
    parser.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--vis_per_type', type=int, default=2)
    parser.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_maps', action='store_true', help='Only the fixed selected samples; default no matrices.')
    parser.add_argument('--analyze_only', action='store_true', help='Saved-data audit only; never import torch or infer.')
    parser.add_argument('--resume_from', help='Reuse verified complete rounds; failed attempts remain untouched.')
    parser.add_argument('--status', help='Display progress only; never start inference.')
    for name in ('_session', '_output'):
        parser.add_argument('--'+name, help=argparse.SUPPRESS)
    parser.add_argument('--_dataset', choices=TARGETS, help=argparse.SUPPRESS)
    parser.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (not args.datasets or args.datasets != sorted(set(args.datasets), key=TARGETS.index)
            or args.shots != sorted(set(args.shots)) or args.num_workers < 0 or args.vis_per_type < 0):
        parser.error('Use ordered unique datasets/shots and nonnegative workers/sample count.')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        parser.error('history_session must be one directory name.')
    internal = [getattr(args, key) is not None for key in ('_session', '_dataset', '_output', '_shot')]
    if any(internal) and not all(internal):
        parser.error('All internal worker arguments required together.')
    if sum(bool(getattr(args, name)) for name in ('status', 'resume_from', 'analyze_only')) > 1:
        parser.error('--status, --resume_from and --analyze_only are separate modes.')
    tokens = list(sys.argv[1:] if argv is None else argv)
    if args.resume_from and any(token.startswith('--') and token.split('=', 1)[0] != '--resume_from' for token in tokens):
        parser.error('Resume restores recorded options; use --resume_from alone.')
    for key in ('new_d_run', 'off_run', 'ablation_run', 'three_run', 'e_run', 'visa_source_root', 'visa_control_root'):
        setattr(args, key, project_path(getattr(args, key), ROOT))
    return args


def history_round(args, dataset, shot, checkpoint_hash, three_hash):
    """Check query/reference/config identity BEFORE any CUDA initialization."""
    source, _, audit = target_inputs(args, dataset, shot, checkpoint_hash)
    relative = Path(dataset)/f'{shot}shot_seed10'
    paths = dict(ablation=args.ablation_run/relative, three=args.three_run/relative, e=args.e_run/relative)
    tables, scores = {}, {}
    refs = [(category, path) for category, path in source['reference_paths']]
    for name, folder in paths.items():
        if read_json(folder/'run_status.json').get('complete') is not True:
            raise ValueError(f'Incomplete history: {folder}')
        if (read_json(folder/'input_audit.json') != audit or read_csv(folder/'sample_manifest.csv') != source['manifest']
                or [(row['Category'], row['Image Path']) for row in read_csv(folder/'references.csv')] != refs):
            raise ValueError(f'Historical input mismatch: {folder}')
        config = read_json(folder/'experiment_config.json')
        if any(config[key] != source['config'][key] for key in (*MODEL_KEYS, 'k_shots')):
            raise ValueError(f'Historical evaluation protocol mismatch: {folder}')
        if name == 'three' and config.get('joint_sha256') != three_hash:
            raise ValueError('Three-control checkpoint identity changed.')
        if name == 'e' and (config.get('checkpoint_sha256') != checkpoint_hash
                            or config.get('rma_fraction') != .125 or config.get('weight_epoch') != 14):
            raise ValueError('E must use unchanged epoch14 / fixed12.5%.')
        table = read_csv(folder/'metrics_combined.csv')
        tables[name] = {(row['Category'], row['Condition']): row for row in table}
        if len(tables[name]) != len(table):
            raise ValueError('Duplicate historical metric row.')
        scores[name] = score_index(read_csv(folder/'image_scores.csv'))
    labels = {(row['Category'], path_key(row['Image Path'])): int(row['Sample Type'] == 'Anomaly')
              for row in source['manifest']}
    identities = set(labels)
    if any(set(index) != identities for index in scores.values()):
        raise ValueError('Historical score query coverage differs.')
    for key in identities:
        if any(int(index[key]['Label']) != labels[key] for index in scores.values()):
            raise ValueError('Historical image label mismatch.')
    mapping = dict(A=('ablation', 'baseline'), T_three=('three', 'new_D_three_control'),
                   D_off=('ablation', 'G0_L0'), G0_L1=('ablation', 'G0_L1'), E=('e', 'dual_output'))
    metrics = []
    for category in [*source['classes'], 'Mean']:
        for name, (kind, condition) in mapping.items():
            row = tables[kind][category, condition]
            if not all(math.isfinite(float(row[field])) for field in FIELDS):
                raise ValueError('Incomplete historical six metrics.')
            metrics.append(dict(Category=category, Condition=name, **{field: float(row[field]) for field in FIELDS},
                                Provenance='historical_saved_full_test', Source=str(paths[kind]/'metrics_combined.csv')))
        for kind in ('three', 'e'):
            if any(abs(float(tables[kind][category, 'baseline'][field])-float(tables['ablation'][category, 'baseline'][field])) > .001 for field in FIELDS):
                raise ValueError('Original-A baseline differs across histories.')
    return source, audit, paths, metrics, scores


def code_hashes():
    names = ('rma_target_diagnostics.py', 'rma_target_diagnostics_utils.py', 'branch_diagnostics.py',
             'branch_diagnostics_utils.py', 'fusion_ablation.py', 'fusion_ablation_utils.py',
             'context_alignment.py', 'rma_reliability.py', 'rma_reliability_utils.py',
             'rma_output_ablation_utils.py', 'rma_fixed_fusion_utils.py', 'rma_local_diagnostics_utils.py',
             'rma_joint_utils.py', 'rma_scratch_common.py', 'rma_utils.py', 'new_d_without_rma.py',
             'rma_model.py', 'rma_new_d_dual_common.py', 'rma_fixed_dual_output.py',
             'rma_beta4_dual_output.py', 'rma_reliability_calibration.py')
    return {name: digest(ROOT/'models_test'/name) for name in names}


def preflight(args):
    """Bind the four existing models and all 18 historical evaluation rounds."""
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    three = read_json(args.three_run/'campaign_config.json')
    three_checkpoint = project_path(three['checkpoint'], ROOT)
    state = read_json(three_checkpoint.parent/'run_status.json')
    manifest = read_json(three_checkpoint.parent/'model_manifest.json')
    if (read_json(args.three_run/'run_status.json').get('complete') is not True
            or three.get('candidate') != 'new_D_three_control' or three.get('flow') != 'new'
            or manifest.get('format') != 'adaptclip-three-scratch-v1'
            or manifest.get('matched_new_flow_three_adapter_control') is not True
            or state.get('complete') is not True or three_checkpoint != project_path(state['selected_checkpoint'], ROOT)
            or digest(three_checkpoint) != three['checkpoint_sha256']):
        raise ValueError('Requires the completed independently trained new-flow three-adapter best checkpoint.')
    e = read_json(args.e_run/'campaign_config.json')
    if (read_json(args.e_run/'run_status.json').get('complete') is not True or e.get('mode') != 'fixed125'
            or e.get('seed') != 10 or e.get('weight_epoch') != 14 or e.get('rma_fraction') != .125
            or e.get('checkpoint_sha256') != checkpoint_hash):
        raise ValueError('Requires the completed experiment E.')
    for name, expected in e['implementation_sha256'].items():
        if digest(ROOT/'models_test'/name) != expected:
            raise ValueError(f'Experiment E implementation changed: {name}')
    new_manifest = read_json(checkpoint.parent/'model_manifest.json')
    for key in ('base_sha256', 'pretrained_sha256', 'source_meta_sha256'):
        if manifest[key] != new_manifest[key]:
            raise ValueError(f'Three/four training input identity differs: {key}')
    for key in ('epochs', 'batch_size', 'accumulation', 'learning_rate', 'betas', 'weight_decay', 'optimizer', 'num_workers',
                'train_reference_shots', 'validation_fraction', 'fusion_weight', 'gradient_clip', 'seed'):
        if manifest['train_options'][key] != new_manifest['train_options'][key]:
            raise ValueError(f'Three/four training setting differs: {key}')
    if read_json(three_checkpoint.parent/'source_split.json') != read_json(checkpoint.parent/'source_split.json'):
        raise ValueError('Three/four training splits differ.')
    files = {str(checkpoint): checkpoint_hash, str(three_checkpoint): three['checkpoint_sha256']}
    inputs, historical, prior_manifest = {}, [], {}
    for dataset in DATASETS:
        inputs[dataset] = {}
        for shot in (1, 2, 4):
            source, audit, paths, metrics, _ = history_round(args, dataset, shot, checkpoint_hash, three['checkpoint_sha256'])
            inputs[dataset][str(shot)] = audit
            historical.extend(dict(Dataset=dataset, Shot=shot, **row) for row in metrics)
            identities = [(row['Category'], path_key(row['Image Path']), row['Sample Type']) for row in source['manifest']]
            if dataset in prior_manifest and identities != prior_manifest[dataset]:
                raise ValueError('Query identities/order must remain fixed across shots.')
            prior_manifest[dataset] = identities
            for folder in paths.values():
                for name in ('run_status.json', 'experiment_config.json', 'input_audit.json',
                             'metrics_combined.csv', 'image_scores.csv', 'references.csv', 'sample_manifest.csv'):
                    path = folder/name
                    files[str(path)] = digest(path)
            for kind in ('source', 'control'):
                for name, value in audit[f'{kind}_hashes'].items():
                    files[str(Path(audit[kind])/name)] = value
            files[str(Path(source['config']['test_data_path'])/'meta.json')] = audit['metadata_sha256']
    for folder in (args.new_d_run, args.off_run, args.ablation_run, args.three_run, args.e_run,
                   checkpoint.parent, three_checkpoint.parent):
        for name in ('campaign_config.json', 'model_manifest.json', 'run_status.json', 'source_split.json'):
            if (folder/name).is_file():
                files[str(folder/name)] = digest(folder/name)
    protocol = new_manifest['base_protocol']
    files[protocol['checkpoint_path']] = new_manifest['base_sha256']
    files[protocol['pretrained_model']] = new_manifest['pretrained_sha256']
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                three_checkpoint=str(three_checkpoint), three_checkpoint_sha256=three['checkpoint_sha256'],
                three_weight_epoch=state['best_epoch'], weight_epoch=14, seed=10,
                implementation_sha256=code_hashes(), historical_sha256=files, inputs=inputs), historical


def write_history(session, rows):
    save_rows(session/'historical_metrics.csv', rows)
    lookup = {(row['Dataset'], row['Shot'], row['Category'], row['Condition']): row for row in rows}
    delta = []
    for dataset in DATASETS:
        for shot in (1, 2, 4):
            for name, after, before in PAIRS:
                a, b = lookup[dataset, shot, 'Mean', after], lookup[dataset, shot, 'Mean', before]
                change = {field: a[field]-b[field] for field in FIELDS}
                delta.append(dict(Dataset=dataset, Shot=shot, Effect=name, After=after, Before=before,
                                  **change, Net=sum(change.values())))
    save_rows(session/'historical_deltas.csv', delta)
    (session/'history_report.md').write_text(
        '# 实验G：已有结果与诊断边界\n\n'
        'historical_metrics.csv保留六数据集18条件的原A、独立三适配器、新D关闭RMA、G0_L1、E完整六指标。\n'
        '这些是历史全测试集结果，不冒充本轮重新计算的像素指标。新D固定第14轮；三适配器对照使用它自己已有的源域最佳权重。\n'
        '后续只对SDD、part_glass补1/2/4-shot中间输出；不训练、不拟合参数、不改变E。\n'
        'D_off权重曾与RMA共同训练，不能当作独立三适配器重训对照。不同训练与最佳轮次的对比不是单一因素因果证明。\n'
        'E图像分数始终来自G0_L1（含25%图最大值的贡献），显示/定位图才是12.5%；两种最大值分列记录。\n'
        '目标标签仅用于误差解释；固定分数切点仅是描述统计，不据此调阈值、融合系数或按数据集开关。\n', encoding='utf-8')


def verify_lock(lock):
    if lock['implementation_sha256'] != code_hashes():
        raise ValueError('Diagnostic/dependency code changed after preflight.')
    for name, value in lock['historical_sha256'].items():
        if digest(name) != value:
            raise ValueError(f'Historical file or checkpoint changed: {name}')


def worker(args):
    # Numerical/project inference imports stay strictly inside the child.
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics import compute_branches
    from branch_diagnostics_utils import evaluate_scores, smooth_maps
    from rma_fixed_fusion_utils import gated_output
    from rma_joint_utils import clone_adapters, encode_learned_text, load_joint_adapters
    from rma_local_diagnostics_utils import summaries
    from rma_output_ablation_utils import route_outputs
    from rma_reliability_utils import load_frozen
    from rma_scratch_common import validate_three_scratch
    from rma_utils import check_frozen, freeze_base
    from test import prompt_association
    from tools import Evaluator
    from tqdm import tqdm

    session, output = project_path(args._session, ROOT), project_path(args._output, ROOT)
    session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
    output.relative_to(session)
    lock = read_json(session/'campaign_config.json')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Worker output must be a fresh empty directory.')
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('experiment-G'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'diagnostics.log', encoding='utf-8'); logger.addHandler(handler)
    state = dict(phase='initializing', complete=False, dataset=args._dataset, shot=args._shot, processed_queries=0)
    write_json(output/'run_status.json', state)
    try:
        verify_lock(lock)
        source, audit, _, history_metrics, history_scores = history_round(
            args, args._dataset, args._shot, lock['checkpoint_sha256'], lock['three_checkpoint_sha256'])
        if audit != lock['inputs'][args._dataset][str(args._shot)]:
            raise ValueError('Scheduled query/reference audit changed.')
        write_json(output/'experiment_config.json', dict(source['config'], experiment='G', training=False,
            new_d_checkpoint=lock['checkpoint'], new_d_epoch=14, three_checkpoint=lock['three_checkpoint'],
            three_epoch=lock['three_weight_epoch'], conditions=CONDITIONS, save_maps=args.save_maps,
            vis_per_type=args.vis_per_type, num_workers=args.num_workers, result_dir=str(output),
            full_pixel_metrics='historical_reused_not_recomputed', threshold_selection=False,
            baseline_verification='G: per-image historical score parity and recalculated image metrics; pixel metrics reused'))
        write_json(output/'input_audit.json', audit)
        save_rows(output/'sample_manifest.csv', source['manifest'])
        write_csv(output/'references.csv', ['Category', 'Image Path'],
                  [dict(Category=category, **{'Image Path': path}) for category, path in source['reference_paths']])
        save_rows(output/'historical_metrics.csv', history_metrics)
        chosen = selected_keys(source['manifest'], args.vis_per_type)
        save_rows(output/'selected_samples.csv', [row for row in source['manifest']
                  if (row['Category'], path_key(row['Image Path'])) in chosen])
        initialized, d_adapters, rma, d_text, snapshots = load_frozen(
            source, output, Path(lock['checkpoint']), args.num_workers, logger)
        config, model, a_t, a_v, a_pq, a_text, memory, loader, dpam, device = initialized
        payload = torch.load(lock['three_checkpoint'], map_location='cpu', weights_only=True)
        validate_three_scratch(payload, digest(config.checkpoint_path), digest(config.pretrained_model), source['config'], 'new')
        if payload['epoch'] != lock['three_weight_epoch']:
            raise ValueError('Three-control selected epoch differs.')
        t_adapters = clone_adapters(initialized)
        load_joint_adapters(t_adapters, payload)
        del payload
        snapshots.append(freeze_base([None, *t_adapters]))
        with torch.no_grad():
            t_text = encode_learned_text(model, t_adapters[0])
        region_rows, gap_rows, effects, score_rows, components, checks, decomposition = [], [], [], [], [], [], []
        count = batches = 0
        visited = set()
        donors = dict(A=('ablation', 'baseline Image Score'), T_three=('three', 'new_D_three_control Image Score'),
                      D_off=('ablation', 'G0_L0 Image Score'), G0_L1=('ablation', 'G0_L1 Image Score'),
                      E=('e', 'dual_output Image Score'))

        def persist():
            for name, rows in (('region_stats.csv', region_rows), ('foreground_background_gaps.csv', gap_rows),
                               ('paired_errors.csv', effects), ('image_scores.csv', score_rows),
                               ('score_components.csv', components), ('historical_score_checks.csv', checks),
                               ('score_decomposition_checks.csv', decomposition)):
                save_rows(output/name, rows)
            summaries(output, region_rows, effects, gap_rows)
            write_json(output/'run_status.json', dict(state, phase='diagnosing', processed_queries=count, encoder_calls=batches))

        with torch.no_grad():
            for items in tqdm(loader, desc=f'G {args._dataset} {args._shot}-shot'):
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf, pp = pf[:, :args._shot].to(device), [value[:, :args._shot].to(device) for value in pp]
                shared = (features, pf, *patches, *pp)
                versions = [value._version for value in shared]
                groups = {}
                for name, adapters, text in (('A', (a_t, a_v, a_pq), a_text),
                                              ('T_three', t_adapters, t_text), ('D_off', d_adapters, d_text)):
                    groups[name] = compute_branches(config, features, patches, pf, pp, *adapters,
                                                    adapters[0].static_text_features, text)
                d_raw, d_maps, d_scores = groups['D_off']
                native = fusion_fun([d_scores[name].to(device) for name in ('VA', 'TA', 'PQA')], fusion_type='average_mean')
                mean3 = fusion_fun([d_raw[name].to(device) for name in ('VA', 'TA', 'PQA')], fusion_type='average_mean')
                old = dict(baseline=d_raw['FINAL'].to(device), no_align=mean3)
                extra = dict(ALIGN=d_raw['ALIGN'].to(device))
                g, local_logits = rma(patches, pp)
                routed_raw, routed_maps, routed_scores, _ = route_outputs(
                    old, {'baseline': d_maps['FINAL']}, {'baseline': d_scores['FINAL']}, native, extra, g, local_logits, config.sigma)
                e_raw, e_map, _, _ = gated_output(old, native, extra, local_logits, None, .125, config.sigma,
                    (routed_raw['G0_L1'], routed_maps['G0_L1'], routed_scores['G0_L1']))
                raw = {name: value[0]['FINAL'] for name, value in groups.items()}
                maps = {name: value[1]['FINAL'] for name, value in groups.items()}
                scores = {name: value[2]['FINAL'] for name, value in groups.items()}
                raw.update(G0_L1=routed_raw['G0_L1'].cpu(), E=e_raw.cpu())
                maps.update(G0_L1=routed_maps['G0_L1'], E=e_map)
                scores.update(G0_L1=routed_scores['G0_L1'], E=routed_scores['G0_L1'])
                native_maps, branch_maps = {}, {}
                for group, (one_raw, one_maps, _) in groups.items():
                    for name in ('VA', 'TA', 'PQA', 'ALIGN'):
                        native_maps[f'{group}_{name}'] = one_raw[name]
                        branch_maps[f'{group}_{name}'] = one_maps[name]
                local = local_logits.softmax(1)[:, 1]
                native_maps.update(RMA_only=local.cpu(), D_mean3=mean3.cpu(),
                                   G0_L1_mean4=((3*mean3+local)/4).cpu(), E_mean125=(.875*mean3+.125*local).cpu())
                branch_maps['RMA_only'] = smooth_maps(local, config.sigma).cpu()
                if versions != [value._version for value in shared]:
                    raise ValueError('Shared CLIP features were mutated.')
                masks = (items['img_mask'][:, 0] > .5).cpu().numpy()
                for i, category in enumerate(items['cls_name']):
                    image, label = items['img_path'][i], int(items['anomaly'][i])
                    expected = source['manifest'][count]
                    if (category != expected['Category'] or path_key(image) != path_key(expected['Image Path'])
                            or ('Anomaly' if label else 'Normal') != expected['Sample Type']):
                        raise ValueError('Query identity/order/label differs from history.')
                    key = category, path_key(image)
                    row = dict(Category=category, **{'Image Path': image}, Label=label)
                    for name in CONDITIONS:
                        value = float(scores[name][i])
                        if not math.isfinite(value):
                            raise ValueError('Nonfinite image score.')
                        kind, field = donors[name]
                        delta = value-float(history_scores[kind][key][field])
                        checks.append(dict(Category=category, Image=image, Condition=name, Delta=delta, Passed=abs(delta) <= 1e-6))
                        if abs(delta) > 1e-6:
                            raise ValueError(f'Historical score changed: {name} {image}, delta={delta}')
                        row[f'{name} Image Score'] = value
                        group = name if name in groups else 'D_off'
                        globals_ = groups[group][2]
                        scoring = maps['G0_L1' if name == 'E' else name][i].numpy()
                        displayed = maps[name][i].numpy()
                        components.append(dict(Category=category, **{'Image Path': image}, Label=label, Condition=name,
                            VA_Global=float(globals_['VA'][i]), TA_Global=float(globals_['TA'][i]), PQA_Global=float(globals_['PQA'][i]),
                            NativeGlobal=float(fusion_fun([globals_[n][i] for n in ('VA', 'TA', 'PQA')], fusion_type='average_mean')),
                            ScoringMapMax=float(scoring.max()), DisplayedMapMax=float(displayed.max()), FinalScore=value,
                            ScoringPeakOnDefect=bool(masks[i].flat[int(scoring.argmax())]),
                            DisplayedPeakOnDefect=bool(masks[i].flat[int(displayed.argmax())]),
                            ScoringMap='G0_L1' if name == 'E' else name))
                        reconstructed = harmonic_score(components[-1]['NativeGlobal'], components[-1]['ScoringMapMax'])
                        difference = reconstructed-value
                        decomposition.append(dict(Category=category, Image=image, Condition=name, Delta=difference,
                                                  Passed=abs(difference) <= 1e-6))
                        if abs(difference) > 1e-6:
                            raise ValueError('Global/map-max components do not reconstruct the actual final score.')
                    score_rows.append(row)
                    stages = {stage: {name: value[i].detach().cpu().numpy() for name, value in values.items()}
                              for stage, values in (('native', native_maps), ('branch_smoothed', branch_maps),
                                                    ('aligned', raw), ('evaluated', maps))}
                    r, gaps, paired = observe(image, category, label, masks[i], stages, args._shot)
                    region_rows.extend(r); gap_rows.extend(gaps); effects.extend(paired)
                    if key in chosen:
                        save_sample(args, output, expected, items['img'][i], masks[i], stages)
                        visited.add(key)
                    count += 1
                batches += 1
                if batches % 10 == 0:
                    persist()
                else:
                    write_json(output/'run_status.json', dict(state, phase='diagnosing', processed_queries=count, encoder_calls=batches))
                del groups, raw, maps, scores, native_maps, branch_maps, routed_raw, routed_maps, routed_scores
                del features, patches, pf, pp, shared, old, extra, local, local_logits, g, mean3, native, e_raw, e_map
        if count != len(source['manifest']) or visited != chosen:
            raise ValueError('Incomplete full-test/selected sample coverage.')
        ranks = ranking_reports(output, score_rows, components)
        metric_lookup = {(row['Category'], row['Condition']): row for row in history_metrics}
        image_checks = []
        for category in source['classes']:
            subset = [row for row in score_rows if row['Category'] == category]
            labels = torch.tensor([row['Label'] for row in subset]).int()
            dummy = torch.zeros((len(subset), 1, 1))
            for name in CONDITIONS:
                evaluator = Evaluator(device, metrics=['I-AUROC', 'I-AP', 'I-F1max'])
                values = evaluate_scores(evaluator, category, dummy, dummy, labels,
                                         torch.tensor([row[f'{name} Image Score'] for row in subset]), device, 'image')
                for short_name, value in values.items():
                    field = 'I-'+short_name
                    delta = value-float(metric_lookup[category, name][field])
                    image_checks.append(dict(Category=category, Condition=name, Metric=field, DeltaPP=delta,
                                             Passed=math.isfinite(delta) and abs(delta) <= .001))
                independent = next(row['AUROC'] for row in ranks if row['Category'] == category
                                   and row['Condition'] == name and row['Component'] == 'FinalScore')
                if independent is None or abs(independent-values['AUROC']) > .001:
                    raise ValueError('Independent pair-ranking AUROC differs from evaluator.')
        save_rows(output/'image_metric_checks.csv', image_checks)
        if not all(row['Passed'] for row in image_checks):
            raise ValueError('Historical image metrics changed.')
        for snapshot in snapshots:
            check_frozen(snapshot)
        verify_lock(lock)
        persist()
        artifact_names = ('historical_metrics.csv', 'region_stats.csv', 'region_summary.csv', 'paired_errors.csv',
                          'foreground_background_gaps.csv', 'score_components.csv', 'image_scores.csv',
                          'historical_score_checks.csv', 'score_decomposition_checks.csv', 'image_metric_checks.csv',
                          'ranking_summary.csv', 'ranking_changed_pairs.csv', 'sample_manifest.csv', 'references.csv')
        artifact_hashes = {name: digest(output/name) for name in artifact_names}
        write_json(output/'run_status.json', dict(state, phase='completed', complete=True, processed_queries=count,
                   encoder_calls=batches, historical_scores_passed=True, image_metrics_passed=True,
                   frozen_integrity_passed=True, selected_samples=len(visited), pixel_metrics_recomputed=False,
                   campaign_checkpoint_sha256=lock['checkpoint_sha256'], implementation_sha256=lock['implementation_sha256'],
                   diagnostic_output_sha256=artifact_hashes))
    except BaseException as error:
        previous = read_json(output/'run_status.json')
        write_json(output/'run_status.json', dict(previous, complete=False, phase='failed', error=f'{type(error).__name__}: {error}'))
        logger.exception('Experiment G stopped.'); raise
    finally:
        cleanup(); handler.close(); logger.removeHandler(handler)


def save_sample(args, output, sample, image, mask, stages):
    """Save only identity-selected samples. Never copy full-test map arrays."""
    import numpy as np
    from fusion_ablation_utils import save_panels
    relative = Path(sample['Category'])/sample['Sample Type']/sample['Sample Key']
    picture = relative.parent/(relative.name+'.png')
    if args.save_visualizations:
        save_panels(output/'visualizations'/picture, image, mask,
                    stages['aligned'], stages['evaluated'], list(CONDITIONS), 3, 'G: frozen final outputs; shared per-image scale')
        for group in ('A', 'T_three', 'D_off'):
            names = [f'{group}_{branch}' for branch in ('VA', 'TA', 'PQA', 'ALIGN')]
            save_panels(output/'branch_visualizations'/group/picture, image, mask,
                        stages['native'], stages['branch_smoothed'], names, 3,
                        'G: individual branch scores, not calibrated final maps')
    if args.save_maps:
        folder = output/'selected_maps'/relative
        folder.mkdir(parents=True, exist_ok=False)
        for stage, maps in stages.items():
            np.savez_compressed(folder/f'{stage}.npz', mask=mask, **maps)


def stored_options(args):
    return {key: str(value) if isinstance(value := getattr(args, key), Path) else value for key in BASE_OPTIONS}


def child_command(args, session, output, dataset, shot):
    command = [sys.executable, str(Path(__file__).resolve()), '--_session', str(session), '--_output', str(output),
               '--_dataset', dataset, '--_shot', str(shot)]
    for key in BASE_OPTIONS:
        value = getattr(args, key)
        if key in ('datasets', 'shots'):
            command.extend(['--'+key, *map(str, value)])
        elif key == 'save_visualizations':
            command.append('--save_visualizations' if value else '--no-save_visualizations')
        elif key == 'save_maps':
            if value:
                command.append('--save_maps')
        else:
            command.extend(['--'+key, str(value)])
    return command


def show_status(session):
    session = project_path(session, ROOT)
    print(f'实验G记录：{session}')
    print(read_json(session/'run_status.json'))
    state = read_json(session/'run_status.json')
    if state.get('current_output') and (Path(state['current_output'])/'run_status.json').is_file():
        print('当前子任务：', read_json(Path(state['current_output'])/'run_status.json'))


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        show_status(args.status); return
    if args._session:
        worker(args); return
    if args.resume_from:
        session = project_path(args.resume_from, ROOT)
        session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
        lock = read_json(session/'campaign_config.json')
        # Resume restores the actual recorded options, not today's CLI defaults.
        for key, value in lock['options'].items():
            setattr(args, key, project_path(value, ROOT) if key.endswith('_run') or key.endswith('_root') else value)
        verify_lock(lock)
        state = read_json(session/'run_status.json')
        if state.get('mode') == 'analyze_only':
            raise ValueError('analyze_only sessions cannot be resumed as inference campaigns.')
    else:
        print('实验G：先核对六数据集18条件的已有结果、参考图与权重。')
        lock, historical = preflight(args)
        session = new_directory(ROOT/'results'/Path(__file__).stem, 'experiment_G_seed10')
        lock['options'] = stored_options(args)
        write_json(session/'campaign_config.json', lock)
        write_history(session, historical)
        state = dict(phase='history_audit_complete', complete=False, mode='analyze_only' if args.analyze_only else 'diagnostics',
                     completed_jobs=[], job_results={}, current=None, current_output=None)
        write_json(session/'run_status.json', state)
        if args.analyze_only:
            write_json(session/'run_status.json', dict(state, complete=True, phase='history_only_completed'))
            print(f'仅已有文件分析完成，没有运行推理：{session}'); return
    jobs = [(dataset, shot) for dataset in args.datasets for shot in args.shots]
    try:
        for dataset, shot in jobs:
            key = f'{dataset}/{shot}'
            prior = state['job_results'].get(key)
            if prior:
                try:
                    child_state = read_json(Path(prior)/'run_status.json')
                except (OSError, ValueError):
                    child_state = {}
                    print(f'保留状态缺失或损坏的未完成尝试，将创建新目录重跑：{prior}')
                if child_state.get('complete') is True:
                    if (child_state.get('campaign_checkpoint_sha256') != lock['checkpoint_sha256']
                            or child_state.get('implementation_sha256') != lock['implementation_sha256']
                            or not all(child_state.get(flag) is True for flag in ('historical_scores_passed', 'image_metrics_passed', 'frozen_integrity_passed'))):
                        raise ValueError('Cannot reuse an unverified completed round.')
                    if not child_state.get('diagnostic_output_sha256'):
                        raise ValueError('Completed round lacks scalar-output integrity evidence.')
                    for name, expected in child_state['diagnostic_output_sha256'].items():
                        if digest(Path(prior)/name) != expected:
                            raise ValueError(f'Completed round output changed: {prior}/{name}')
                    if key not in state['completed_jobs']:
                        state['completed_jobs'].append(key)
                    print(f'复用已完成诊断：{key}'); continue
            output = new_directory(session/dataset, f'{shot}shot_seed10')
            state.update(phase='diagnostic_inference', current=key, current_output=str(output), complete=False)
            state['job_results'][key] = str(output)
            write_json(session/'run_status.json', state)
            print(f'[{jobs.index((dataset, shot))+1}/{len(jobs)}] {dataset} {shot}-shot；独立子进程，结束后释放缓存。', flush=True)
            command = child_command(args, session, output, dataset, shot)
            write_json(session/'last_command.json', dict(job=key, output=str(output), command=command))
            run_child(command)
            if read_json(output/'run_status.json').get('complete') is not True:
                raise ValueError('Child exited without successful diagnostic completion.')
            cleanup()
            state['completed_jobs'].append(key)
            write_json(session/'run_status.json', state)
        verify_lock(lock)
        merge_rounds(session, state)
        state.update(phase='completed', complete=True, current=None, current_output=None)
        state.pop('error', None)
        write_json(session/'run_status.json', state)
        print(f'实验G诊断完成：{session}')
    except BaseException as error:
        state.update(complete=False, error=f'{type(error).__name__}: {error}')
        write_json(session/'run_status.json', state)
        print(f"实验G中断；阶段={state['phase']}，当前={state['current']}，已完成={state['completed_jobs']}\n记录：{session}\n子任务：{state['current_output']}", flush=True)
        raise


def merge_rounds(session, state):
    """Combine scalar summaries only; no pixel matrices and no new selection."""
    for name in ('region_summary.csv', 'paired_error_summary.csv', 'gap_summary.csv',
                 'ranking_summary.csv', 'image_metric_checks.csv'):
        combined = []
        for job, folder in state['job_results'].items():
            dataset, shot = job.split('/')
            path = Path(folder)/name
            if path.is_file():
                combined.extend(dict(Dataset=dataset, Shot=int(shot), **row) for row in read_csv(path))
        save_rows(session/name, combined)
    (session/'README_results.md').write_text(
        '# 实验G已完成\n\n'
        '历史完整六指标见historical_metrics.csv、historical_deltas.csv。\n'
        '新生成的region_summary、gap_summary解释缺陷与背景响应；ranking_summary解释各分数的排序能力。\n'
        '各轮ranking_changed_pairs、ranking_error_images用于找出新增/恢复的排序错误；score_components明确区分评分图与显示图最大值。\n'
        'image_metric_checks核对本轮图像三指标；historical_score_checks逐图核对历史分数。像素完整指标复用历史，未在本轮重算。\n'
        '每个job_results指向实际成功的轮次目录；中断尝试保留，不混入汇总。\n'
        '原A、T_three、D_off、G0_L1、E分别表示原三适配器、独立新流程三适配器、共同训练后关闭RMA、25%图评分、12.5%定位双输出。\n'
        '区域均值并非越高越好；需要同时看缺陷与背景。不同分支分数不可仅凭热图颜色比较。\n'
        '不从目标标签选阈值/比例，不自动决定新算法；源域最佳轮次差异及单seed限制仍须披露。\n', encoding='utf-8')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'实验G未正常完成：{type(error).__name__}: {error}', file=sys.stderr)
        raise
