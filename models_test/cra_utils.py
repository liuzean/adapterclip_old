"""CRA source/weight contracts and shared validation/target evaluation.

Top-level imports are standard-library-only; schedulers never initialize CUDA.
"""
import math
from collections import Counter, defaultdict
from pathlib import Path

from fusion_ablation_utils import path_key, read_csv, read_json, write_csv
from h_inference import WEIGHT_SHA
from rma_align_ablation_utils import metric_index, verified_history
from rma_control_io import write_json
from rma_utils import FIELDS, cleanup, digest

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('H', 'K_CRA')
FORMAT = 'adaptclip-cra-only-v1'
SPEC = dict(dim=768, layers=4, kernel=3)


def code_hashes():
    from rma_align_ablation import code_hashes as h_hashes
    names = ('cra_model.py', 'cra_utils.py', 'train_cra.py', 'test_cra.py', 'cra_workflow.py',
             'cra_pipeline.py', 'run_align_cra_sequence.py', 'rma_control_resume.py', 'rma_reliability_utils.py')
    return dict(h_hashes(), **{str(ROOT/'models_test'/n): digest(ROOT/'models_test'/n) for n in names})


def verify_plan(plan):
    if plan['code_sha256'] != code_hashes():
        raise ValueError('CRA/H inference or training implementation changed during the campaign.')
    for path, value in plan['input_sha256'].items():
        if digest(path) != value:
            raise ValueError('CRA source/weight/history input changed: '+path)


def source_plan(checkpoint):
    """Use the saved new-D source split and episodes; do not sample new references."""
    from rma_reliability_utils import source_inputs
    checkpoint = Path(checkpoint).resolve()
    if digest(checkpoint) != WEIGHT_SHA:
        raise ValueError('CRA must start from the fixed H/new-D epoch14 checkpoint.')
    source, split, audit = source_inputs(checkpoint)
    folder = checkpoint.parent
    manifest, state = read_json(folder/'model_manifest.json'), read_json(folder/'run_status.json')
    if state.get('complete') is not True or state.get('best_epoch') != 14:
        raise ValueError('H source training is incomplete or has changed.')
    settings = manifest['train_options']
    required = dict(epochs=15, batch_size=2, accumulation=4, learning_rate=.0001,
                    weight_decay=.0001, betas=[.9, .999], optimizer='adamw',
                    train_reference_shots=[1, 2, 4], fusion_weight=1., gradient_clip=1., num_workers=0)
    if any(settings[k] != v for k, v in required.items()):
        raise ValueError('Saved new-D settings differ from the intended CRA warm-start protocol.')
    if read_json(folder/'training_queries.json') != split['train_queries']:
        raise ValueError('Saved source training query order differs.')
    if read_json(folder/'training_reference_pool.json') != split['training_support_pool']:
        raise ValueError('Saved normal reference pool differs.')
    root = Path(source['config']['test_data_path'])
    queries = {path_key(root/r['img_path']): r for r in split['train_queries']}
    supports = {r['img_path']: r for pool in split['training_support_pool'].values() for r in pool}
    episodes = read_csv(folder/'training_episodes.csv')
    grouped = defaultdict(list)
    for row in episodes:
        epoch, step, shot = (int(row[k]) for k in ('epoch', 'step', 'shot'))
        if epoch not in range(1, 16) or shot not in (1, 2, 4):
            raise ValueError('Invalid source episode epoch/shot.')
        query = queries.get(path_key(row['query']))
        refs = row['references'].split(' | ')
        if query is None or query['cls_name'] != row['category'] or len(refs) != shot:
            raise ValueError('Source episode query/category/reference count differs.')
        if any(r not in supports or supports[r]['cls_name'] != row['category'] or int(supports[r]['anomaly']) != 0 for r in refs):
            raise ValueError('Source episode contains a non-normal or wrong-class reference.')
        grouped[epoch, step].append(row)
    for epoch in range(1, 16):
        batches = [grouped[e, s] for e, s in sorted(grouped) if e == epoch]
        if (len(batches) != 690 or Counter(path_key(r['query']) for b in batches for r in b) != Counter(queries.keys())
                or [s for e, s in sorted(grouped) if e == epoch] != list(range(1, 691))):
            raise ValueError('Source episode coverage/order/batch budget differs from new D.')
        if any(len(b) != 2 or len({r['shot'] for r in b}) != 1 or int(b[0]['shot']) != (1, 2, 4)[(i % 3)]
               for i, b in enumerate(batches)):
            raise ValueError('Source batch or 1/2/4 reference cycle differs.')
    paths = [checkpoint, root/'meta.json', Path(source['config']['pretrained_model']),
             Path(source['config']['checkpoint_path'])]
    paths += [folder/n for n in ('model_manifest.json', 'run_status.json', 'source_split.json',
                               'training_queries.json', 'training_reference_pool.json', 'training_episodes.csv')]
    return source, split, grouped, dict(input_sha256={str(p): digest(p) for p in paths},
        code_sha256=code_hashes(), source_audit=audit, training_settings=required,
        h_checkpoint=str(checkpoint), h_checkpoint_sha256=WEIGHT_SHA, source_root=str(root))


def checked_checkpoint(path):
    """Read completed CRA selection receipts without loading torch or model weights."""
    path = Path(path).resolve()
    path.parent.relative_to((ROOT/'results/train_cra').resolve())
    plan, status = read_json(path.parent/'campaign_config.json'), read_json(path.parent/'run_status.json')
    verify_plan(plan)
    if (path.name != 'best.pth' or status.get('complete') is not True
            or not all(status.get(k) is True for k in ('frozen_integrity_passed', 'gradient_checks_passed', 'zero_init_parity_passed'))
            or status.get('completed_epochs') != 15 or digest(path) != status.get('selected_checkpoint_sha256')):
        raise ValueError('Requires this CRA experiment\'s completed source-selected best.pth.')
    history = read_csv(path.parent/'training_history.csv')
    if (digest(path.parent/'training_history.csv') != status.get('training_history_sha256')
            or digest(path.parent/'source_H_metrics.csv') != status.get('source_H_sha256')
            or [int(r['epoch']) for r in history] != list(range(16))
            or any(not math.isfinite(float(r['validation_score'])) for r in history)):
        raise ValueError('CRA source history must include epoch0 and all15 trained epochs.')
    best = max(history, key=lambda r: float(r['validation_score']))
    if int(best['epoch']) != status['best_epoch'] or abs(float(best['validation_score'])-status['best_score']) > 1e-9:
        raise ValueError('CRA selected epoch differs from source-only validation history.')
    return plan, status, digest(path)


def attach(state, checkpoint=None):
    import torch
    from cra_model import ContextResidualAggregation, PQAResidualHook
    with torch.random.fork_rng(devices=[]):
        # CPU Conv initialization only; torch.manual_seed would also reseed CUDA.
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(10).get_state())
        cra = ContextResidualAggregation(**SPEC).to(state.device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        selected = read_json(Path(checkpoint).parent/'run_status.json')
        if (payload.get('format') != FORMAT or payload.get('h_checkpoint_sha256') != WEIGHT_SHA or payload.get('cra_spec') != SPEC
                or payload.get('epoch') != selected['best_epoch'] or abs(payload['validation_score']-selected['best_score']) > 1e-9):
            raise ValueError('Wrong CRA weight format/base/interface.')
        cra.load_state_dict(payload['cra'], strict=True)
    return cra, PQAResidualHook(state.adapters[2], cra)


def predict_pair(state, hook, images, classes, paths):
    """Encode once; freeze RMA/ALIGN; apply H formulas to both H and CRA outputs.

    Returns final maps [B,H,W], image scores [B], and scalar SE/G0 components.
    Unlike experiment J, candidate image scores may change through modified PQA.
    """
    import torch
    from fusion_ablation import infer_batch
    from h_inference import select_score
    from rma_fixed_fusion_utils import gated_output
    from rma_output_ablation_utils import route_outputs
    from test import prompt_association
    features, patches = state.model.encode_image(images.to(state.device), state.config.features_list, DPAM_layer=20)
    pf, pp = prompt_association(*state.memory, classes)
    pf, pp = pf.to(state.device), [p.to(state.device) for p in pp]
    shared = (features, pf, *patches, *pp); versions = [v._version for v in shared]
    g, local = state.rma(patches, pp)
    maps, scores, parts, alignment = {}, {}, {}, None
    for name in CONDITIONS:
        before = hook.calls
        with hook.active(name == 'K_CRA'):
            old, old_maps, old_scores, _, _, native, extra = infer_batch(
                state.config, features, patches, pf, pp, *state.adapters, state.text, paths)
        if hook.calls-before != (4 if name == 'K_CRA' else 0):
            raise ValueError('CRA did not run exactly once at each intended PQA layer.')
        if alignment is None:
            alignment = extra['ALIGN']
        elif not torch.equal(alignment, extra['ALIGN']):
            raise ValueError('CRA changed the frozen reference ALIGN path.')
        raw, routed, scored, _ = route_outputs(old, old_maps, old_scores, native, extra, g, local, state.config.sigma)
        _, maps[name], _, _ = gated_output(old, native, extra, local, None, .125, state.config.sigma,
            (raw['G0_L1'], routed['G0_L1'], scored['G0_L1']))
        scores[name] = select_score(scored['G0_L1'], native.detach().cpu(), 'H')
        parts[name] = dict(SE=scored['G0_L1'], G0=native.detach().cpu())
    if versions != [v._version for v in shared]:
        raise ValueError('CRA mutated shared query/reference features.')
    return maps, scores, parts


def evaluate(state, hook, loader, output, *, expected, history_scores=None, zero_init=False, progress=None):
    """Full category metrics, one-category RAM buffer, shared source/target logic."""
    import torch
    from rma_utils import MetricsBuffer, save_tables
    from tqdm import tqdm
    pixels, images, score_rows, checks = [], [], [], []
    buffer, count, closed = None, 0, set()
    def flush():
        p, i = buffer.finish(state.device); pixels.extend(p); images.extend(i)
        save_tables(output, pixels, images); cleanup()
    with torch.no_grad():
        for step, items in enumerate(tqdm(loader, desc=str(output.name)), 1):
            if progress: progress(step)
            maps, scores, parts = predict_pair(state, hook, items['img'], items['cls_name'], items['img_path'])
            if zero_init and (not torch.equal(maps['H'], maps['K_CRA']) or not torch.equal(scores['H'], scores['K_CRA'])):
                raise ValueError('Zero-initialized CRA does not exactly restore H predictions.')
            if any(m.shape != items['img_mask'][:, 0].shape or not torch.isfinite(m).all() for m in maps.values()):
                raise ValueError('Invalid CRA output map shape/value.')
            for i, category in enumerate(items['cls_name']):
                if count >= len(expected): raise ValueError('Extra query in evaluation.')
                path, label = items['img_path'][i], int(items['anomaly'][i]); row = expected[count]
                if (category != row['Category'] or path_key(path) != path_key(row['Image Path'])
                        or label != int(row['Sample Type'] == 'Anomaly')):
                    raise ValueError('CRA query order/identity/label changed.')
                if buffer is None or category != buffer.category:
                    if buffer is not None: flush(); closed.add(buffer.category)
                    if category in closed: raise ValueError('Noncontiguous category.')
                    buffer = MetricsBuffer(category, conditions=CONDITIONS)
                buffer.append(items['img_mask'][i, 0] > .5, label, maps, scores, i)
                record = dict(Category=category, **{'Image Path': path}, Label=label)
                for name in CONDITIONS:
                    record[name] = float(scores[name][i])
                    record[name+'_SE'] = float(parts[name]['SE'][i]); record[name+'_G0'] = float(parts[name]['G0'][i])
                score_rows.append(record)
                if history_scores is not None:
                    delta = record['H']-float(history_scores[count]['H Image Score'])
                    check = dict(Image=path, Delta=delta, Passed=math.isfinite(delta) and abs(delta) <= 1e-6)
                    checks.append(check)
                    if not check['Passed']:
                        write_csv(output/'score_checks.csv', list(check), checks)
                        raise ValueError('CRA-off H scores disagree with historical H.')
                count += 1
            del maps, scores, parts
    if buffer is None or count != len(expected): raise ValueError('Incomplete CRA evaluation.')
    flush()
    table = save_tables(output, pixels, images)
    write_csv(output/'image_scores.csv', list(score_rows[0]), score_rows)
    if checks: write_csv(output/'score_checks.csv', list(checks[0]), checks)
    return table


def compare(current, history, classes):
    fresh = metric_index(current, CONDITIONS, classes)
    old = metric_index(history, ('H', 'baseline'), classes)
    checks, deltas = [], []
    for category in (*classes, 'Mean'):
        h, candidate = fresh[category, 'H'], fresh[category, 'K_CRA']
        for f in FIELDS:
            value = float(h[f])-float(old[category, 'H'][f])
            checks.append(dict(Category=category, Metric=f, DeltaPP=value, Passed=abs(value) <= .001))
        for name, row in (('H', h), ('baseline', old[category, 'baseline'])):
            change = {f:float(candidate[f])-float(row[f]) for f in FIELDS}
            deltas.append(dict(Category=category, Condition='K_CRA', Comparator=name, **change,
                ImageNet=sum(change[f] for f in FIELDS[:3]), PixelNet=sum(change[f] for f in FIELDS[3:]), Net=sum(change.values())))
    return list(current)+[r for r in history if r['Condition'] == 'baseline'], checks, deltas
