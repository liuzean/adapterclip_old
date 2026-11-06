"""Frozen-reference cohesion gate and audits; no torch import in schedulers."""
import math
from pathlib import Path

from fusion_ablation_utils import read_csv, read_json, write_csv, write_json
from rma_utils import FIELDS, MetricsBuffer, digest, source_split

ROOT = Path(__file__).resolve().parents[1]
RULE = 'normal-top8-pairwise-dispersion-v1'
BETAS = (0., 4., 16., 64.)
CANDIDATE = 'G0_L1_reliable'
CONDITIONS = ('baseline', 'G0_L1', 'G1_L1', CANDIDATE)
SELECTION = 'Maximize source category-macro six-metric mean across 1/2/4-shot; beta=0 first on ties.'


def dispersion_map(queries, supports, image_size, *, top_k=8, chunk=128):
    """Matching ambiguity, NOT defect probability, [B,H,W] in [0,2].

    Four projected query [B,1+L,D] / reference [B,S,1+L,D] layers. Reuse
    RMA's frozen cosine Top-8 retrieval definition (extra chunked retrieval,
    no extra encoder pass). For its unit reference vectors r_i, dispersion is
    mean_{i!=j}(1-r_i dot r_j) = K/(K-1)*(1-||mean_i r_i||^2).
    Average four patch-grid dispersions, then interpolate like the RMA head.
    It also works for 1-shot: eight patches of ONE normal image, no extra image.
    """
    import torch
    from torch.nn import functional as F
    if len(queries) != 4 or len(supports) != 4 or top_k != 8 or min(chunk, image_size) < 1:
        raise ValueError('Requires four layers, fixed Top-8 and positive sizes.')
    layers = []
    with torch.no_grad():
        for query, support in zip(queries, supports):
            b, tokens, width = query.shape
            if (support.ndim != 4 or support.shape[0] != b or support.shape[-2:] != (tokens, width)
                    or support.shape[1] < 1 or width != 768):
                raise ValueError('Expected projected 768-D query/support interface.')
            q = F.normalize(query[:, 1:].detach().float(), dim=-1)
            r = F.normalize(support[:, :, 1:].detach().float().reshape(b, -1, width), dim=-1)
            side = math.isqrt(tokens-1)
            if side*side != tokens-1 or r.shape[1] < top_k:
                raise ValueError('Square patch grid and at least eight reference patches required.')
            if not torch.isfinite(q).all() or not torch.isfinite(r).all():
                raise ValueError('Nonfinite matching feature.')
            if (r.norm(dim=-1) < .99).any() or (q.norm(dim=-1) < .99).any():
                raise ValueError('Zero/degenerate matching features cannot express reliability.')
            batch = torch.arange(b, device=q.device)[:, None, None]
            parts = []
            for start in range(0, tokens-1, chunk):
                index = (q[:, start:start+chunk] @ r.transpose(1, 2)).topk(top_k, dim=-1).indices
                center = r[batch, index].mean(2)
                parts.append(((1-center.square().sum(-1))*top_k/(top_k-1)).clamp(0, 2))
            layers.append(torch.cat(parts, dim=1).reshape(b, 1, side, side))
        result = F.interpolate(torch.stack(layers).mean(0), (image_size, image_size),
                               mode='bilinear', align_corners=True)[:, 0].clamp(0, 2)
    return result


def gated_output(old, fixed, extra, local_logits, dispersion, beta, sigma, endpoint):
    """Only local mixture changes. Return raw/map/final image score and alpha.

    alpha=.25*exp(-beta*dispersion); other three maps keep equal proportions.
    Mix BEFORE unchanged ALIGN harmonic fusion and Gaussian smoothing. Native
    global remains new-D three-adapter mean G0; image score uses OWN map max.
    beta=0 aliases exact historical G0_L1 arithmetic, avoiding rounding drift.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import smooth_maps
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0 or dispersion.shape != local_logits[:, 0].shape:
        raise ValueError('Invalid gate strength/map shape.')
    if not torch.isfinite(dispersion).all() or (dispersion < 0).any() or (dispersion > 2).any():
        raise ValueError('Invalid matching dispersion.')
    alpha = .25*torch.exp(-beta*dispersion)
    if beta == 0:
        return (*endpoint, alpha)
    local = local_logits.softmax(1)[:, 1]
    mixed = (1-alpha)*old['no_align'] + alpha*local
    raw = fusion_fun([mixed, extra['ALIGN']], fusion_type='harmonic_mean')
    maps = smooth_maps(raw, sigma).cpu()
    maximum = maps.flatten(1).max(1).values.to(fixed.device)
    score = fusion_fun([fixed, maximum], fusion_type='harmonic_mean').cpu()
    return raw, maps, score, alpha


class ReliabilityBuffer(MetricsBuffer):
    """At most three target map copies; full new D shares G0_L1 pixel output."""
    def __init__(self, category, conditions=CONDITIONS):
        super().__init__(category, tuple(n for n in conditions if n != 'G1_L1'))
        self.full_scores = []

    def append(self, mask, label, maps, scores, index):
        import torch
        if maps['G1_L1'] is not maps['G0_L1'] and not torch.equal(maps['G1_L1'], maps['G0_L1']):
            raise ValueError('G0_L1 and full new D must have identical pixel maps.')
        super().append(mask, label, maps, scores, index)
        if not scores['G1_L1'][index].isfinite().all():
            raise ValueError('Nonfinite full-new-D image score.')
        self.full_scores.append(scores['G1_L1'][index].detach().cpu().clone())

    def finish(self, device):
        import torch
        from branch_diagnostics_utils import evaluate_scores
        from tools import Evaluator
        labels = torch.tensor(self.labels).int()
        pixels, images = super().finish(device)
        pixels.append(dict(next(r for r in pixels if r['Condition'] == 'G0_L1'), Condition='G1_L1'))
        dummy = torch.zeros((len(labels), 1, 1))
        evaluator = Evaluator(device, metrics=[f'I-{m}' for m in ('AUROC', 'AP', 'F1max')])
        images.append(dict(Category=self.category, Condition='G1_L1', **evaluate_scores(
            evaluator, self.category, dummy, dummy, labels, torch.stack(self.full_scores), device, 'image')))
        self.full_scores.clear()
        return pixels, images


def source_inputs(checkpoint):
    """Read ONLY source training metadata; reproduce existing held-out split."""
    training = Path(checkpoint).parent
    manifest = read_json(training/'model_manifest.json')
    config = dict(manifest['base_protocol'], k_shots=4)
    if config['dataset'] != 'mvtec' or config['seed'] != 10 or config['batch_size'] != 2:
        raise ValueError('Requires the existing new-D MVTec seed10/batch2 validation protocol.')
    root = Path(config['test_data_path'])
    if digest(root/'meta.json') != manifest['source_meta_sha256']:
        raise ValueError('Source metadata changed since new-D training.')
    split = read_json(training/'source_split.json')
    if split != source_split(root, 10, manifest['validation_fraction_used']):
        raise ValueError('New-D held-out split or normal reference identities changed.')
    if digest(config['checkpoint_path']) != manifest['base_sha256'] or digest(config['pretrained_model']) != manifest['pretrained_sha256']:
        raise ValueError('CLIP/original-A weight identity changed.')
    source = dict(config=config, classes=split['classes'], references=split['validation_supports'])
    audit = dict(meta_sha256=digest(root/'meta.json'), split_sha256=digest(training/'source_split.json'),
                 manifest_sha256=digest(training/'model_manifest.json'), query_count=len(split['validation_queries']),
                 reference_count=len(split['validation_supports']), batch_size=2, seed=10, shots=[1, 2, 4])
    return source, split, audit


def implementation_hashes():
    names = ('rma_reliability.py', 'rma_reliability_calibration.py', 'rma_reliability_utils.py',
             'rma_output_ablation_utils.py', 'rma_model.py', 'fusion_ablation.py', 'rma_utils.py',
             'rma_joint_utils.py', 'fusion_ablation_utils.py', 'branch_diagnostics_utils.py')
    return {n: digest(ROOT/'models_test'/n) for n in names}


def load_frozen(source, output, checkpoint, num_workers, logger):
    """Reuse original initialization; load/freeze the SAME four new-D states."""
    import torch
    from fusion_ablation import initialize
    from rma_joint_utils import clone_adapters, encode_learned_text, load_joint_adapters
    from rma_model import ReferenceMatchingAdapter
    from rma_scratch_common import validate_scratch
    from rma_utils import freeze_base
    from tools import setup_seed
    setup_seed(10)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = source['config']
    validate_scratch(payload, digest(config['checkpoint_path']), digest(config['pretrained_model']), config, 'new')
    if payload['epoch'] != 14 or payload['model_spec']['top_k'] != 8 or payload['model_spec']['matching'] != 'learned':
        raise ValueError('Requires unchanged new-D epoch14/learned Top-8 RMA.')
    initialized = initialize(source, output, num_workers, logger)
    snapshots = [freeze_base(initialized)]
    adapters = clone_adapters(initialized)
    load_joint_adapters(adapters, payload)
    rma = ReferenceMatchingAdapter(**payload['model_spec']).to(initialized[-1]).eval()
    rma.load_state_dict(payload['rma'], strict=True)
    snapshots.append(freeze_base([None, *adapters, rma]))
    with torch.no_grad():
        text = encode_learned_text(initialized[1], adapters[0])
    return initialized, adapters, rma, text, snapshots


def verify_calibration(directory, checkpoint, checkpoint_hash):
    """Reject unlabeled/edited/other-weight calibration; NEVER fit on targets."""
    directory = Path(directory)
    locked = read_json(directory/'locked_config.json')
    _, _, audit = source_inputs(checkpoint)
    if (read_json(directory/'run_status.json').get('complete') is not True
            or locked['rule'] != RULE or locked['betas'] != list(BETAS)
            or locked['checkpoint_sha256'] != checkpoint_hash or locked['weight_epoch'] != 14
            or locked['seed'] != 10 or locked['source_audit'] != audit
            or locked['implementation_sha256'] != implementation_hashes() or locked['selection'] != SELECTION):
        raise ValueError('Calibration is incomplete or weights/source/rule/implementation changed.')
    tables = {}
    for shot in (1, 2, 4):
        child = directory/f'{shot}shot_seed10'
        if read_json(child/'run_status.json').get('complete') is not True:
            raise ValueError('Source calibration round incomplete.')
        if digest(child/'metrics_combined.csv') != locked['metric_sha256'][str(shot)]:
            raise ValueError('Source calibration metrics changed after locking.')
        tables[shot] = read_csv(child/'metrics_combined.csv')
    beta, _ = select_beta(tables)
    if beta != locked['beta']:
        raise ValueError('Locked beta does not match the predeclared source selection.')
    return locked


def metric_checks(output, table, directory, pairs, filename='endpoint_metric_checks.csv'):
    current = {(r['Category'], r['Condition']): r for r in table}
    checks = []
    for now, before in pairs:
        previous = [r for r in read_csv(directory/'metrics_combined.csv') if r['Condition'] == before]
        if {r['Category'] for r in previous} != {c for c, n in current if n == now}:
            raise ValueError('Historical metric category coverage differs.')
        for row in previous:
            for field in FIELDS:
                delta = float(current[row['Category'], now][field])-float(row[field])
                checks.append(dict(Category=row['Category'], Condition=now, Metric=field,
                                   Delta=delta, Passed=math.isfinite(delta) and abs(delta) <= .001))
    write_csv(output/filename, list(checks[0]), checks)
    if not all(r['Passed'] for r in checks):
        raise ValueError(f'Historical endpoint metrics changed: {directory}')


def select_beta(tables):
    """One strength shared by all classes, datasets AND shots; zero wins ties."""
    rows = []
    for beta in BETAS:
        name = 'G0_L1' if beta == 0 else f'beta_{int(beta)}'
        values = [float(next(r for r in tables[s] if r['Category'] == 'Mean' and r['Condition'] == name)[f])
                  for s in (1, 2, 4) for f in FIELDS]
        if len(values) != 18 or not all(math.isfinite(v) for v in values):
            raise ValueError('Incomplete/nonfinite source validation metrics.')
        rows.append(dict(Beta=beta, Score=sum(values)/len(values)))
    # Suppress only numerical ties (1e-9 percentage points).
    best = rows[0]
    for row in rows[1:]:
        if row['Score'] > best['Score'] + 1e-9:
            best = row
    return best['Beta'], rows


def summarize(session, jobs, datasets, *, candidate=CANDIDATE):
    combined, deltas = [], []
    for dataset, shot in jobs:
        values = read_csv(session/dataset/f'{shot}shot_seed10'/'metrics_combined.csv')
        combined.extend(dict(Dataset=dataset, Shot=shot, **r) for r in values)
        means = {r['Condition']: r for r in values if r['Category'] == 'Mean'}
        for comparator in ('G0_L1', 'baseline', 'G1_L1'):
            change = {f: float(means[candidate][f])-float(means[comparator][f]) for f in FIELDS}
            deltas.append(dict(Dataset=dataset, Shot=shot, Condition=candidate, Comparator=comparator,
                               **change, Net=sum(change.values())))
    for filename, mean in [('metrics_summary.csv', True), ('metrics_by_category.csv', False)]:
        write_csv(session/filename, ['Dataset', 'Shot', 'Category', 'Condition', *FIELDS],
                  [r for r in combined if (r['Category'] == 'Mean') == mean])
    retention, overall = [], []
    for comparator, suffix in [('G0_L1', 'G0_L1'), ('baseline', 'original_A'), ('G1_L1', 'new_D')]:
        group = [r for r in deltas if r['Comparator'] == comparator]
        write_csv(session/f'deltas_vs_{suffix}.csv', list(group[0]), group)
        one = []
        for shot in sorted({s for _, s in jobs}):
            rows = [r for r in group if r['Shot'] == shot]
            positive = sum(r['Net'] > 0 for r in rows)
            net = sum(r['Net'] for r in rows)
            complete = set(datasets) == {'Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass'} and {r['Dataset'] for r in rows} == set(datasets)
            one.append(dict(Comparator=comparator, Shot=shot, PositiveDatasets=positive, Net=net,
                            SixDatasetsComplete=complete, Pass=bool(complete and positive >= 3 and net > 0)))
        retention.extend(one)
        complete = len(one) == 3 and {r['Shot'] for r in one} == {1, 2, 4} and all(r['SixDatasetsComplete'] for r in one)
        overall.append(dict(Comparator=comparator, Primary=comparator == 'G0_L1', All18Complete=complete,
                            PassingShots=sum(r['Pass'] for r in one), Retain=bool(complete and sum(r['Pass'] for r in one) >= 2)))
    write_csv(session/'retention_by_shot.csv', list(retention[0]), retention)
    write_json(session/'retention_overall.json', overall)
