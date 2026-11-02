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

    def __init__(self, category):
        self.category = category
        self.masks, self.labels = [], []
        self.maps = {n: [] for n in CONDITIONS}
        self.scores = {n: [] for n in CONDITIONS}

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.cpu().to(dtype=torch.uint8).clone())
        self.labels.append(int(label))
        for name in CONDITIONS:
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
        for name in CONDITIONS:
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
    write_json(Path(output)/'run_status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt)
               else 'failed', complete=False, error=str(error)))
