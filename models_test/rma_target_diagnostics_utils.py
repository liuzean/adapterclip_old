"""Experiment G: bounded region statistics and tie-aware image ranking audits.

Scheduler-safe imports. Target masks explain errors; they never fit a rule.
"""
import bisect
import math
from collections import defaultdict

from fusion_ablation_utils import path_key, write_csv
from rma_local_diagnostics_utils import THRESHOLDS, describe, regions

CONDITIONS = ('A', 'T_three', 'D_off', 'G0_L1', 'E')
PAIRS = (('training_flow', 'T_three', 'A'), ('joint_training', 'D_off', 'T_three'),
         ('add_local_25', 'G0_L1', 'D_off'), ('candidate_vs_off', 'E', 'D_off'),
         ('candidate_vs_A', 'E', 'A'))


def save_rows(path, rows):
    if rows:
        write_csv(path, list(rows[0]), rows)


def score_index(rows):
    """Reject duplicate queries instead of silently replacing a score row."""
    result = {}
    for row in rows:
        key = row['Category'], path_key(row['Image Path'])
        if key in result:
            raise ValueError(f'Duplicate historical score identity: {key}')
        result[key] = row
    return result


def selected_keys(manifest, per_type):
    """First N per class/label in the original full manifest, never by score."""
    counts = defaultdict(int)
    selected = set()
    for row in manifest:
        group = row['Category'], row['Sample Type']
        if counts[group] < per_type:
            selected.add((row['Category'], path_key(row['Image Path'])))
        counts[group] += 1
    return selected


def observe(image, category, label, mask, stages, shot):
    """Reduce each [H,W] map immediately; retain only scalar statistics.

    Stages may contain branch scores, pre-ALIGN means or final maps. Comparisons
    are within a stage. Fixed .25/.5/.75 cuts are descriptive, not operating
    thresholds or calibrated probabilities. Empty regions stay undefined.
    """
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    selections = regions(mask, label)
    identity = dict(Shot=shot, Category=category, Image=image, Label=label)
    rows, gaps, effects = [], [], []
    for stage, maps in stages.items():
        for condition, value in maps.items():
            value = np.asarray(value)
            if value.shape != mask.shape:
                raise ValueError('Diagnostic map and GT shape differ.')
            describe(value)
            for region, selected in selections.items():
                data = value[selected]
                stats = describe(data)
                extra = {}
                for threshold in THRESHOLDS:
                    high = int((data > threshold).sum())
                    extra[f'HighCount_{threshold}'] = high
                    extra[f'HighRate_{threshold}'] = high / len(data) if len(data) else None
                    extra[f'MissRate_{threshold}'] = (len(data)-high)/len(data) if len(data) and region == 'defect' else None
                rows.append(dict(identity, Stage=stage, Condition=condition, Region=region, **stats, **extra))
            if label:
                fg, bg = value[mask], value[~mask]
                gaps.append(dict(identity, Stage=stage, Condition=condition, DefectPixels=len(fg), BackgroundPixels=len(bg),
                                 ForegroundMinusBackground=float(fg.mean()-bg.mean()) if len(fg) and len(bg) else None,
                                 ForegroundMeanMinusBackgroundP99=float(fg.mean()-np.quantile(bg, .99)) if len(fg) and len(bg) else None))
        for name, after, before in PAIRS:
            if after not in maps or before not in maps:
                continue
            for region, selected in selections.items():
                a, b = np.asarray(maps[after])[selected], np.asarray(maps[before])[selected]
                n = len(a)
                for threshold in THRESHOLDS:
                    new = int(((a > threshold) & (b <= threshold)).sum())
                    removed = int(((a <= threshold) & (b > threshold)).sum())
                    effects.append(dict(identity, Stage=stage, Effect=name, After=after, Before=before, Region=region,
                                        Threshold=threshold, Pixels=n, DeltaSum=float((a.astype('float64')-b).sum()),
                                        MeanDelta=float((a.astype('float64')-b).mean()) if n else None,
                                        NewHighPixels=new, RemovedHighPixels=removed,
                                        NewMissPixels=removed if region == 'defect' else None,
                                        RecoveredPixels=new if region == 'defect' else None))
    return rows, gaps, effects


def pair_credit(anomaly, normal):
    return 1. if anomaly > normal else .5 if anomaly == normal else 0.


def rank_summary(rows, field):
    """AUROC is the fraction of correctly ordered anomaly/normal pairs.

    Ties contribute half; a single-label set is undefined, never AUROC=0.
    """
    if any(int(row['Label']) not in (0, 1) for row in rows):
        raise ValueError('Expected binary image labels.')
    values = [float(row[field]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite image ranking score.')
    normal = sorted(float(row[field]) for row in rows if int(row['Label']) == 0)
    anomaly = [float(row[field]) for row in rows if int(row['Label']) == 1]
    tied = correct = inverted = 0
    for value in anomaly:
        left, right = bisect.bisect_left(normal, value), bisect.bisect_right(normal, value)
        correct += left
        tied += right-left
        inverted += len(normal)-right
    total = len(normal)*len(anomaly)
    return dict(NormalImages=len(normal), AnomalyImages=len(anomaly), Pairs=total,
                CorrectPairs=correct, TiedPairs=tied, InvertedPairs=inverted,
                AUROC=100*(correct+.5*tied)/total if total else None)


def harmonic_score(native, map_max):
    """Scalar diagnostic reconstruction; never replaces inference outputs."""
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in (native, map_max)):
        raise ValueError('Expected finite probability-score components in [0,1].')
    return 2*native*map_max/(native+map_max) if native+map_max else 0.


def ranking_reports(output, score_rows, components):
    """Full-test ranking changes and per-image responsibility; no thresholds.

    Changed pairs are post-hoc diagnostic examples, not a test subset selected
    to estimate performance. No labels are used to select model parameters.
    """
    groups = defaultdict(list)
    for row in score_rows:
        groups[row['Category']].append(row)
    summaries, changes, contributions = [], [], []
    for category, rows in groups.items():
        for condition in CONDITIONS:
            summaries.append(dict(Category=category, Condition=condition, Component='FinalScore',
                                  **rank_summary(rows, f'{condition} Image Score')))
        for name, after, before in PAIRS:
            impacts = defaultdict(lambda: [0., 0.])
            for anomaly in (row for row in rows if int(row['Label']) == 1):
                for normal in (row for row in rows if int(row['Label']) == 0):
                    af, bf = f'{after} Image Score', f'{before} Image Score'
                    old = pair_credit(float(anomaly[bf]), float(normal[bf]))
                    new = pair_credit(float(anomaly[af]), float(normal[af]))
                    if old == new:
                        continue
                    delta = new-old
                    changes.append(dict(Category=category, Effect=name, Before=before, After=after,
                                        AnomalyImage=anomaly['Image Path'], NormalImage=normal['Image Path'],
                                        BeforeAnomalyScore=anomaly[bf], BeforeNormalScore=normal[bf],
                                        AfterAnomalyScore=anomaly[af], AfterNormalScore=normal[af],
                                        PairCreditDelta=delta))
                    for row in (anomaly, normal):
                        impacts[row['Image Path']][0] += max(0., -delta)
                        impacts[row['Image Path']][1] += max(0., delta)
            for image, (lost, recovered) in impacts.items():
                contributions.append(dict(Category=category, Effect=name, Image=image,
                                          LostPairCredit=lost, RecoveredPairCredit=recovered, NetLost=lost-recovered))
    component_groups = defaultdict(list)
    for row in components:
        component_groups[row['Category'], row['Condition']].append(row)
    for (category, condition), rows in component_groups.items():
        for field in ('VA_Global', 'TA_Global', 'PQA_Global', 'NativeGlobal', 'ScoringMapMax'):
            summaries.append(dict(Category=category, Condition=condition, Component=field, **rank_summary(rows, field)))
    save_rows(output/'ranking_summary.csv', summaries)
    # Include a header even when no pair changed, so absence is unambiguous.
    fields = ('Category', 'Effect', 'Before', 'After', 'AnomalyImage', 'NormalImage',
              'BeforeAnomalyScore', 'BeforeNormalScore', 'AfterAnomalyScore', 'AfterNormalScore', 'PairCreditDelta')
    write_csv(output/'ranking_changed_pairs.csv', fields, changes)
    save_rows(output/'ranking_error_images.csv', sorted(contributions, key=lambda row: row['NetLost'], reverse=True))
    return summaries
