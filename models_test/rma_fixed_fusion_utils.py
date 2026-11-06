"""One source-selected constant local fraction; no spatial gate or retraining."""
import math
from pathlib import Path

from fusion_ablation_utils import read_csv, read_json, write_csv, write_json
from rma_reliability_utils import implementation_hashes as shared_hashes
from rma_reliability_utils import source_inputs
from rma_utils import FIELDS, digest

RULE = 'rma-constant-local-fraction-v1'
PARAMETER = 'rma_fraction'
CANDIDATE = 'G0_L1_fixed'
CONDITIONS = ('baseline', 'G0_L1', 'G1_L1', CANDIDATE)
# Existing 25% wins numerical ties. Other candidates sorted by increasing value.
CHOICES = ((.25, 'G0_L1'), (0., 'fraction_0'), (.125, 'fraction_125'), (.375, 'fraction_375'))
SELECTION = 'Maximize source category-macro six-metric mean across 1/2/4-shot; keep25% on ties within1e-9pp.'


def dispersion_map(queries, supports, image_size, *, chunk=128):
    """No spatial confidence is calculated in this control."""
    return None


def gated_output(old, fixed, extra, local_logits, dispersion, fraction, sigma, endpoint):
    """Constant alpha for ALL pixels/classes/shots; change only pre-ALIGN mix.

    Inputs/returns match shared runner. Three-map mean [B,H,W], RMA logits
    [B,2,H,W]. alpha0 restores new-D three-map path; alpha.25 EXACT G0_L1.
    Final image score always fuses G0 and this condition's own smoothed max.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import smooth_maps
    fraction = float(fraction)
    if not math.isfinite(fraction) or fraction not in {x for x, _ in CHOICES} or dispersion is not None:
        raise ValueError('Only the predeclared constant fractions are allowed; no spatial gate.')
    alpha = torch.full_like(local_logits[:, 0], fraction)
    if fraction == .25:
        return (*endpoint, alpha)
    if fraction == 0:
        raw = old['baseline']
    else:
        mixed = (1-fraction)*old['no_align'] + fraction*local_logits.softmax(1)[:, 1]
        raw = fusion_fun([mixed, extra['ALIGN']], fusion_type='harmonic_mean')
    maps = smooth_maps(raw, sigma).cpu()
    maximum = maps.flatten(1).max(1).values.to(fixed.device)
    score = fusion_fun([fixed, maximum], fusion_type='harmonic_mean').cpu()
    return raw, maps, score, alpha


def implementation_hashes():
    root = Path(__file__).resolve().parent
    return dict(shared_hashes(), **{n: digest(root/n) for n in ('rma_fixed_fusion.py', 'rma_fixed_fusion_utils.py')})


def select_fraction(tables):
    rows = []
    for fraction, name in CHOICES:
        values = []
        for shot in (1, 2, 4):
            means = [r for r in tables[shot] if r['Category'] == 'Mean' and r['Condition'] == name]
            if len(means) != 1:
                raise ValueError('Missing/duplicate source macro row.')
            values.extend(float(means[0][f]) for f in FIELDS)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite source calibration metrics.')
        rows.append(dict(RMAFraction=fraction, Condition=name, Score=sum(values)/18))
    best = rows[0]
    for row in rows[1:]:
        if row['Score'] > best['Score'] + 1e-9:
            best = row
    return best['RMAFraction'], rows


def lock(directory, checkpoint, checkpoint_hash):
    tables = {}
    for shot in (1, 2, 4):
        child = directory/f'{shot}shot_seed10'
        state = read_json(child/'run_status.json')
        if not all(state.get(k) is True for k in ('complete', 'source_endpoint_parity_passed', 'frozen_integrity_passed')):
            raise ValueError('All source rounds must finish and pass.')
        tables[shot] = read_csv(child/'metrics_combined.csv')
    fraction, rows = select_fraction(tables)
    write_csv(directory/'source_selection.csv', list(rows[0]), rows)
    write_json(directory/'locked_config.json', dict(rule=RULE, choices=CHOICES, rma_fraction=fraction,
        selection=SELECTION, checkpoint_sha256=checkpoint_hash, weight_epoch=14, seed=10,
        source_audit=source_inputs(checkpoint)[2], implementation_sha256=implementation_hashes(),
        metric_sha256={str(s): digest(directory/f'{s}shot_seed10/metrics_combined.csv') for s in (1, 2, 4)},
        target_labels_used=False, target_evaluation_required=fraction != .25,
        note='Same source validation used for epoch14 selection; not independent unbiased performance.'))
    write_json(directory/'run_status.json', dict(status='completed', complete=True, rma_fraction=fraction))
    return fraction


def verify_calibration(directory, checkpoint, checkpoint_hash):
    directory = Path(directory); locked = read_json(directory/'locked_config.json')
    if (read_json(directory/'run_status.json').get('complete') is not True or locked['rule'] != RULE
            or locked['choices'] != [list(x) for x in CHOICES] or locked['selection'] != SELECTION
            or locked['checkpoint_sha256'] != checkpoint_hash or locked['weight_epoch'] != 14 or locked['seed'] != 10
            or locked['source_audit'] != source_inputs(checkpoint)[2]
            or locked['implementation_sha256'] != implementation_hashes()):
        raise ValueError('Incomplete/stale/other-weight fixed-fraction calibration.')
    tables = {}
    for shot in (1, 2, 4):
        path = directory/f'{shot}shot_seed10'
        state = read_json(path/'run_status.json')
        if not all(state.get(k) is True for k in ('complete', 'source_endpoint_parity_passed', 'frozen_integrity_passed')):
            raise ValueError('Incomplete source round.')
        if digest(path/'metrics_combined.csv') != locked['metric_sha256'][str(shot)]:
            raise ValueError('Source results modified after locking.')
        tables[shot] = read_csv(path/'metrics_combined.csv')
    if select_fraction(tables)[0] != locked['rma_fraction']:
        raise ValueError('Locked fraction does not match source-only selection.')
    return locked
