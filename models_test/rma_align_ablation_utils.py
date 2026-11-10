"""Experiment J: change ALIGN on the localization path only; no training."""
import math
from collections import Counter

from fusion_ablation_utils import path_key, read_csv, read_json
from rma_utils import FIELDS, digest

CONDITIONS = ('H', 'J_no_ALIGN')
DATASETS = ('Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass')


def local_outputs(three_map, local_logits, align, sigma):
    """Return raw/final [B,H,W] pairs with the SAME 12.5% mixture and filter.

    three_map is the new-D VA/TA/PQA pre-ALIGN mean, NOT original A.
    The only difference is harmonic(mixed, ALIGN) versus mixed itself.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import smooth_maps
    if (sigma != 4 or three_map.ndim != 3 or align.shape != three_map.shape
            or local_logits.shape != (three_map.shape[0], 2, *three_map.shape[1:])):
        raise ValueError('Requires shared [B,H,W] maps, [B,2,H,W] logits and sigma4.')
    mixed = .875 * three_map + .125 * local_logits.softmax(1)[:, 1]
    raw = dict(H=fusion_fun([mixed, align], fusion_type='harmonic_mean'), J_no_ALIGN=mixed)
    maps = {name: smooth_maps(value, sigma).cpu() for name, value in raw.items()}
    if any(not torch.isfinite(value).all() for value in (*raw.values(), *maps.values())):
        raise ValueError('Nonfinite ALIGN-control map; no sanitizing or rescaling permitted.')
    return raw, maps


def predict(state, images, classes, paths):
    """One encoder/old-three/RMA pass; both conditions alias one fresh H score.

    H score = .75 * SE + .25 * G0. SE retains the historical 25% G0_L1
    scoring path. Neither the 12.5% map nor the no-ALIGN map rescales it.
    """
    import torch
    from fusion_ablation import infer_batch
    from h_inference import LOCAL_FRACTION, select_score
    from rma_fixed_fusion_utils import gated_output
    from rma_output_ablation_utils import route_outputs
    from test import prompt_association
    features, patches = state.model.encode_image(images.to(state.device), state.config.features_list, DPAM_layer=20)
    pf, pp = prompt_association(*state.memory, classes)
    shared = (features, pf, *patches, *pp)
    versions = [value._version for value in shared]
    old, old_maps, old_scores, _, _, native, extra = infer_batch(
        state.config, features, patches, pf, pp, *state.adapters, state.text, paths)
    global_logits, local_logits = state.rma(patches, pp)
    routed_raw, routed_maps, routed_scores, _ = route_outputs(
        old, old_maps, old_scores, native, extra, global_logits, local_logits, state.config.sigma)
    _, reference_pixel, _, _ = gated_output(
        old, native, extra, local_logits, None, LOCAL_FRACTION, state.config.sigma,
        (routed_raw['G0_L1'], routed_maps['G0_L1'], routed_scores['G0_L1']))
    _, maps = local_outputs(old['no_align'], local_logits, extra['ALIGN'], state.config.sigma)
    if not torch.equal(maps['H'], reference_pixel):
        raise ValueError('Fresh ALIGN-on pixel map differs from the existing H/E12.5% path.')
    e_score = routed_scores['G0_L1']
    g0 = native.detach().cpu()
    h_score = select_score(e_score, g0, 'H')
    if not torch.isfinite(h_score).all():
        raise ValueError('Nonfinite H image score.')
    if versions != [value._version for value in shared]:
        raise ValueError('Shared CLIP/reference features mutated.')
    # Reuse the same tensor, so changing localization cannot alter image scores.
    return maps, {name: h_score for name in CONDITIONS}, dict(E=e_score, G0=g0)


def metric_index(rows, conditions, categories):
    expected = {(c, n) for c in (*categories, 'Mean') for n in conditions}
    index = {(r['Category'], r['Condition']): r for r in rows}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError('Incomplete/duplicate metric category or condition coverage.')
    if any(not math.isfinite(float(row[f])) or not 0 <= float(row[f]) <= 100
           for row in rows for f in FIELDS):
        raise ValueError('Invalid six-metric values.')
    return index


def verified_history(inputs):
    """Verify the historical table/scores BEFORE constructing a CUDA model."""
    folder = inputs['history']
    state = read_json(folder/'run_status.json')
    for name in ('metrics_combined.csv', 'image_scores.csv'):
        expected = state.get('H_output_sha256', {}).get(name)
        if not expected or digest(folder/name) != expected:
            raise ValueError(f'Historical H output changed or lacks a fingerprint: {name}')
    table = [r for r in read_csv(folder/'metrics_combined.csv') if r['Condition'] in ('H', 'baseline')]
    metric_index(table, ('H', 'baseline'), inputs['classes'])
    scores = read_csv(folder/'image_scores.csv')
    if len(scores) != len(inputs['manifest']):
        raise ValueError('Historical score coverage differs from query manifest.')
    for row, query in zip(scores, inputs['manifest']):
        value = float(row['H Image Score'])
        if (row['Category'] != query['Category'] or path_key(row['Image Path']) != path_key(query['Image Path'])
                or int(row['Label']) != int(query['Sample Type'] == 'Anomaly')
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Historical score query identity/label/value differs.')
    if inputs['config']['fusion_type'] != 'average_mean':
        raise ValueError('Requires unchanged original three-branch averaging.')
    return table, scores


def comparison_tables(current, history, categories):
    """H endpoint parity, exact image invariance, and all J-H/J-A differences."""
    fresh = metric_index(current, CONDITIONS, categories)
    previous = metric_index(history, ('H', 'baseline'), categories)
    parity, identity, deltas = [], [], []
    for category in (*categories, 'Mean'):
        h, j = fresh[category, 'H'], fresh[category, 'J_no_ALIGN']
        for field in FIELDS:
            delta = float(h[field])-float(previous[category, 'H'][field])
            parity.append(dict(Category=category, Metric=field, DeltaPP=delta, Passed=abs(delta) <= .001))
        for field in FIELDS[:3]:
            delta = float(j[field])-float(h[field])
            identity.append(dict(Category=category, Metric=field, DeltaPP=delta, Passed=delta == 0))
        for name, old in (('H', h), ('baseline', previous[category, 'baseline'])):
            change = {f: float(j[f])-float(old[f]) for f in FIELDS}
            deltas.append(dict(Category=category, Condition='J_no_ALIGN', Comparator=name, **change,
                ImageNet=sum(change[f] for f in FIELDS[:3]), PixelNet=sum(change[f] for f in FIELDS[3:]),
                Net=sum(change.values())))
    combined = list(current)+[r for r in history if r['Condition'] == 'baseline']
    return combined, parity, identity, deltas


def retention(deltas):
    """Full six-dataset, per-shot rule; incomplete campaigns cannot be retained."""
    rows = []
    for comparator in ('H', 'baseline'):
        for shot in (1, 2, 4):
            part = [r for r in deltas if r['Category'] == 'Mean' and r['Comparator'] == comparator and int(r['Shot']) == shot]
            complete = Counter(r['Dataset'] for r in part) == Counter(DATASETS)
            improved = sum(float(r['Net']) > 0 for r in part)
            gains = sum(max(float(r['Net']), 0) for r in part)
            losses = -sum(min(float(r['Net']), 0) for r in part)
            rows.append(dict(Comparator=comparator, Shot=shot, Complete=complete, Improved=improved,
                Gain=gains, Loss=losses, Net=gains-losses, Pass=complete and improved >= 3 and gains > losses))
    overall = {}
    for comparator in ('H', 'baseline'):
        part = [r for r in rows if r['Comparator'] == comparator]
        complete = all(r['Complete'] for r in part)
        passes = sum(r['Pass'] for r in part)
        overall[comparator] = dict(complete=complete, passing_shots=passes, retain=complete and passes >= 2)
    return rows, overall
