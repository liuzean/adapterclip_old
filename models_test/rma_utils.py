"""RMA experiment plumbing; scheduler imports are standard-library only."""
import gc
import hashlib
import math
import os
import random
import subprocess
from pathlib import Path

from fusion_ablation_utils import read_json, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max')
CONDITIONS = ('baseline', 'rma')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def run_child(command):
    """Wait for exit, including on interruption, before any next GPU job."""
    with subprocess.Popen(command, cwd=str(ROOT)) as child:
        try:
            code = child.wait()
        except KeyboardInterrupt:
            try:
                child.wait(timeout=10)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if child.poll() is None:
                    child.terminate()
                child.wait()
            raise
    if code:
        raise RuntimeError(f'Child process failed with exit code {code}; remaining jobs stopped.')


def cleanup():
    """Release unused allocations; process exit releases live tensors/context."""
    import sys
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def atomic_checkpoint(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def freeze_base(initialized):
    """Freeze parameters AND running buffers of CLIP/VA/TA/PQA."""
    modules = initialized[1:5]
    for module in modules:
        module.eval()
        module.requires_grad_(False)
    parameters = [(p, p._version) for module in modules for p in module.parameters()]
    buffers = [(b, b.detach().cpu().clone()) for module in modules for b in module.buffers()]
    return modules, parameters, buffers


def check_frozen(snapshot, optimizer=None):
    import torch
    modules, parameters, buffers = snapshot
    if any(m.training for module in modules for m in module.modules()):
        raise RuntimeError('A frozen module was switched to training mode.')
    if any(p.requires_grad or p.grad is not None or p._version != v for p, v in parameters):
        raise RuntimeError('Frozen parameters changed or acquired gradients.')
    if any(not torch.equal(b.detach().cpu(), previous) for b, previous in buffers):
        raise RuntimeError('Frozen running buffers changed.')
    if optimizer is not None:
        old_ids = {id(p) for p, _ in parameters}
        if any(id(p) in old_ids for group in optimizer.param_groups for p in group['params']):
            raise RuntimeError('Optimizer includes an original parameter.')


def fuse_rma(old_raw, extras, fixed_global, global_logits, local_logits, sigma=None):
    """Four-way average then original ALIGN/image harmonic formulas.

    sigma=None returns differentiable PRE-smoothing map and native image mean
    for training. Validation/test use unchanged SciPy smoothing and max score.
    """
    from adaptcliplib import fusion_fun
    local = local_logits.softmax(1)[:, 1]
    global_score = global_logits.softmax(-1)[:, 1]
    raw = fusion_fun([(3 * old_raw['no_align'] + local) / 4, extras['ALIGN']],
                     fusion_type='harmonic_mean')
    native = (3 * fixed_global + global_score) / 4
    if sigma is None:
        return raw, native
    from branch_diagnostics_utils import smooth_maps
    maps = smooth_maps(raw, sigma).cpu()
    maximum = maps.flatten(1).max(1).values.to(native.device)
    score = fusion_fun([native, maximum], fusion_type='harmonic_mean').cpu()
    return raw, maps, score


class MetricsBuffer:
    """One category on host RAM, original evaluator, independent conditions."""

    def __init__(self, category, conditions=CONDITIONS):
        self.category = category
        self.conditions = tuple(conditions)
        self.masks, self.labels = [], []
        self.maps = {n: [] for n in self.conditions}
        self.scores = {n: [] for n in self.conditions}

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.cpu().to(dtype=torch.uint8).clone())
        self.labels.append(int(label))
        for name in self.conditions:
            if not maps[name][index].isfinite().all() or not scores[name][index].isfinite().all():
                raise ValueError('Nonfinite evaluated output.')
            self.maps[name].append(maps[name][index].detach().cpu().clone())
            self.scores[name].append(scores[name][index].detach().cpu().clone())

    def finish(self, device):
        import torch
        from branch_diagnostics_utils import evaluate_scores
        from tools import Evaluator
        masks, labels = torch.stack(self.masks).int(), torch.tensor(self.labels).int()
        pixels, images = [], []
        for name in self.conditions:
            maps, scores = torch.stack(self.maps[name]), torch.stack(self.scores[name])
            self.maps[name].clear()
            evaluator = Evaluator(device, metrics=[f'P-{n}' for n in ('AUROC', 'AP', 'F1max')])
            pixels.append(dict(Category=self.category, Condition=name, **evaluate_scores(
                evaluator, self.category, masks, maps, labels, scores, device, 'pixel')))
            del maps, evaluator
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f'I-{n}' for n in ('AUROC', 'AP', 'F1max')])
            images.append(dict(Category=self.category, Condition=name, **evaluate_scores(
                evaluator, self.category, dummy, dummy, labels, scores, device, 'image')))
            self.scores[name].clear()
        self.masks.clear()
        self.labels.clear()
        return pixels, images


def source_split(root, seed, fraction):
    """Source-only stratified held-out queries; supports from normal train rows.

    The original Dataset uses meta.test as labeled source queries. We explicitly
    hold out a subset of those queries from NEW RMA gradient training.
    """
    meta = read_json(Path(root) / 'meta.json')
    rng = random.Random(seed)
    train, validation, references = [], [], []
    classes = list(meta['test'])
    supports = {}
    for category in classes:
        supports[category] = [r for r in meta['train'][category] if int(r['anomaly']) == 0]
        if len(supports[category]) < 4:
            raise ValueError(f'{category}: need at least four distinct normal supports.')
        chosen = rng.sample(supports[category], 4)
        references.extend(chosen)
        for label in (0, 1):
            rows = [r for r in meta['test'][category] if int(r['anomaly']) == label]
            if len(rows) < 2:
                raise ValueError(f'{category}: need train and validation queries for label {label}.')
            rng.shuffle(rows)
            count = max(1, min(len(rows)-1, round(len(rows)*fraction)))
            validation.extend(rows[:count])
            train.extend(rows[count:])
    def identities(rows):
        return {str((Path(root)/r['img_path']).resolve()).casefold() for r in rows}
    if identities(train) & identities(validation):
        raise ValueError('Source train/validation query overlap.')
    all_supports = [r for group in supports.values() for r in group]
    for label, rows in (('train queries', train), ('validation queries', validation), ('supports', all_supports)):
        if len(identities(rows)) != len(rows):
            raise ValueError(f'Duplicate source {label}.')
        for row in rows:
            image = Path(root)/row['img_path']
            image.resolve().relative_to(Path(root).resolve())
            if not image.is_file():
                raise FileNotFoundError(image)
            if row['cls_name'] not in classes:
                raise ValueError('Unknown source category.')
            if int(row['anomaly']) and not (Path(root)/row['mask_path']).is_file():
                raise FileNotFoundError(Path(root)/row['mask_path'])
    if identities(all_supports) & (identities(train) | identities(validation)):
        raise ValueError('Source support/query overlap.')
    return dict(classes=classes, train_queries=train, validation_queries=validation,
                validation_supports=references, training_support_pool=supports,
                validation_shots=[1, 2, 4], seed=seed, validation_fraction=fraction)


def source_spec(args, split):
    """Copy stable model protocol; replace only source dataset, batch and shot."""
    config = read_json(ROOT/'results/branch_diagnostics/visa_1shot_seed10/experiment_config.json')
    config.update(dataset='mvtec', test_data_path=str(args.train_data_path),
                  pretrained_model=str(args.pretrained_model), checkpoint_path=str(args.base_checkpoint),
                  batch_size=args.batch_size, k_shots=4, seed=10)
    return dict(config=config, classes=split['classes'], references=split['validation_supports'])


def validation_score(combined_by_shot):
    """Predeclared equal mean: category -> six metrics -> 1/2/4 shots."""
    scores = []
    for shot in (1, 2, 4):
        rows = [r for r in combined_by_shot[shot] if r['Category'] == 'Mean' and r['Condition'] == 'rma']
        if len(rows) != 1 or any(rows[0][f] is None or not math.isfinite(rows[0][f]) for f in FIELDS):
            raise ValueError('Undefined source validation selection score.')
        scores.append(sum(rows[0][f] for f in FIELDS)/len(FIELDS))
    return sum(scores)/len(scores)


def curves(output, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ('global_loss', 'local_loss', 'fusion_loss', 'total_loss'):
        axes[0].plot([r['epoch'] for r in history], [r[name] for r in history], label=name)
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[1].plot([r['epoch'] for r in history], [r['validation_score'] for r in history])
    axes[1].set(xlabel='Epoch', ylabel='Source full-model validation score (0-100)')
    fig.tight_layout()
    try:
        fig.savefig(Path(output)/'training_curves.png', dpi=140)
    finally:
        plt.close(fig)


def save_tables(output, pixels, images):
    from fusion_ablation_utils import metric_tables
    rows = metric_tables(output, pixels, images)
    write_csv(Path(output)/'metrics_combined.csv', ['Category', 'Condition', *FIELDS], rows)
    return rows


def validate_checkpoint_payload(payload, base_hash, pretrained_hash, protocol):
    if payload.get('format') != 'adaptclip-rma-v1':
        raise ValueError('Unsupported RMA checkpoint format.')
    if payload['base_sha256'] != base_hash or payload['pretrained_sha256'] != pretrained_hash:
        raise ValueError('RMA requires the same frozen base weights used in training.')
    keys = ('features_list', 'image_size', 'n_ctx', 'vl_reduction', 'pq_mid_dim', 'pq_context',
            'visual_learner', 'textual_learner', 'pq_learner', 'fusion_type', 'sigma')
    if any(payload['base_protocol'][key] != protocol[key] for key in keys):
        raise ValueError('RMA training and target base model protocols differ.')
    if payload['model_spec']['input_dim'] != 768 or payload['model_spec']['image_size'] != protocol['image_size']:
        raise ValueError('RMA feature/image interface differs from the base protocol.')


def failure_status(output, error):
    # Preserve the last epoch/dataset/shot when marking a failure.
    path = Path(output)/'run_status.json'
    try:
        previous = read_json(path)
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError):
        previous = {}
    previous.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                    complete=False, error=str(error))
    write_json(path, previous)


def record_pipeline_progress(path, phase, **details):
    """Optional exact-session receipt for an outer scheduler; no model changes."""
    if path is None:
        return
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, dict(phase=phase, **details))
    temporary.replace(path)


def compare_training_runs(old_run, new_run, output):
    """Offline comparison of two completed B runs, with identical target protocol.

    New minus old answers the training-settings question. New minus original A
    remains the separate algorithm baseline comparison in the test session.
    """
    from fusion_ablation_utils import read_csv
    datasets = ('Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass')
    old_run, new_run, output = Path(old_run), Path(new_run), Path(output)
    if old_run.resolve() == new_run.resolve():
        raise ValueError('Training comparison requires distinct old and new runs.')
    campaigns = [read_json(r/'campaign_config.json') for r in (old_run, new_run)]
    manifests = []
    for root, campaign in zip((old_run, new_run), campaigns):
        if read_json(root/'run_status.json').get('complete') is not True:
            raise ValueError('Both comparison test runs must be complete.')
        manifests.append(read_json(Path(campaign['checkpoint']).parent/'model_manifest.json'))
    for key in ('base_sha256', 'pretrained_sha256', 'source_split_sha256', 'model_spec'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f'Training comparison differs in fixed model/data field: {key}')
    # Training manifests store seed beside train_options, including legacy B runs.
    if manifests[0]['seed'] != manifests[1]['seed']:
        raise ValueError('Training comparison differs in fixed model/data field: seed')
    for key in ('validation_fraction', 'fusion_weight', 'matching', 'hidden_dim', 'top_k'):
        if manifests[0]['train_options'][key] != manifests[1]['train_options'][key]:
            raise ValueError(f'Unexpected training change beyond settings profile: {key}')
    if campaigns[0]['datasets'] != campaigns[1]['datasets'] or campaigns[0]['shots'] != campaigns[1]['shots']:
        raise ValueError('Training comparison requires the same ordered target dataset/shot scope.')
    rows = []
    for dataset in campaigns[1]['datasets']:
        for shot in campaigns[1]['shots']:
            dirs = [r/dataset/f'{shot}shot_seed10' for r in (old_run, new_run)]
            configs = [read_json(d/'experiment_config.json') for d in dirs]
            for key in ('seed','k_shots','dataset','test_data_path','pretrained_model','checkpoint_path',
                        'image_size','features_list','sigma','batch_size','n_ctx','vl_reduction','pq_mid_dim','pq_context',
                        'fusion_type','visual_learner','textual_learner','pq_learner','disable_rma'):
                if configs[0][key] != configs[1][key]:
                    raise ValueError(f'Target evaluation protocol changed: {dataset}/{shot}/{key}')
            for filename in ('references.csv', 'sample_manifest.csv'):
                if read_csv(dirs[0]/filename) != read_csv(dirs[1]/filename):
                    raise ValueError(f'Historical target identities differ: {filename}')
            for d in dirs:
                state = read_json(d/'run_status.json')
                if state.get('complete') is not True or state.get('baseline_parity_passed') is not True:
                    raise ValueError('Incomplete round or original-A parity failure.')
            tables = [{(r['Category'],r['Condition']):r for r in read_csv(d/'metrics_combined.csv')} for d in dirs]
            if set(tables[0]) != set(tables[1]):
                raise ValueError('Category/condition sets differ.')
            for category, condition in tables[1]:
                if condition == 'baseline':
                    if any(float(tables[0][category,condition][f]) != float(tables[1][category,condition][f]) for f in FIELDS):
                        raise ValueError('Original-A outputs differ between the two training tests.')
                    continue
                if condition != 'rma':
                    raise ValueError('Only two B/RMA-only runs can be compared here.')
                delta = {f:float(tables[1][category,condition][f])-float(tables[0][category,condition][f]) for f in FIELDS}
                rows.append(dict(Dataset=dataset, Shot=shot, Category=category, **delta, Net=sum(delta.values())))
    output.mkdir(parents=True, exist_ok=False)
    for filename, is_mean in (('new_minus_old_by_dataset.csv',True),('new_minus_old_by_category.csv',False)):
        write_csv(output/filename,['Dataset','Shot','Category',*FIELDS,'Net'],[r for r in rows if (r['Category']=='Mean')==is_mean])
    summary = []
    for shot in campaigns[1]['shots']:
        group=[r for r in rows if r['Shot']==shot and r['Category']=='Mean']
        gain=sum(max(r['Net'],0) for r in group)
        loss=-sum(min(r['Net'],0) for r in group)
        positive=sum(r['Net']>0 for r in group)
        complete={r['Dataset'] for r in group}==set(datasets)
        summary.append(dict(Shot=shot, PositiveDatasets=positive, Gains=gain, Losses=loss,
                            Net=gain-loss, SixDatasetsComplete=complete, Retain=bool(complete and positive>=3 and gain>loss)))
    write_csv(output/'new_minus_old_retention.csv',list(summary[0]),summary)
    write_json(output/'comparison_config.json',dict(old_test_run=str(old_run),new_test_run=str(new_run),
        old_training_options=manifests[0]['train_options'],new_training_options=manifests[1]['train_options'],
        original_A_comparison='See new test session deltas_vs_original.csv and retention_by_shot.csv',
        interpretation='Seven training settings changed together; does not isolate the cause to a single parameter.'))
