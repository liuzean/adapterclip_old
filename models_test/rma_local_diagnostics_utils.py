"""Bounded per-image local error statistics; imports numerical packages lazily."""
import math
from collections import defaultdict

from fusion_ablation_utils import write_csv

CONDITIONS = ('baseline', 'G0_L0', 'RMA_only', 'G0_L1')
THRESHOLDS = (.25, .5, .75)
PAIRS = (('three_vs_A', 'G0_L0', 'baseline'), ('add_RMA', 'G0_L1', 'G0_L0'))


def regions(mask, label):
    """Disjoint source GT regions; never invent defect pixels on normal images."""
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or label not in (0, 1) or (label == 0 and mask.any()):
        raise ValueError('Expected2D binary GT; normal images must have empty masks.')
    return {'normal_background': ~mask} if label == 0 else {'anomaly_background': ~mask, 'defect': mask}


def describe(values):
    """Exact within-image region tails; missing region is None, not zero."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or (values < -1e-6).any() or (values > 1+1e-6).any():
        raise ValueError('Map scores must be finite and within probability-score bounds.')
    n = len(values)
    if not n:
        return dict(Pixels=0, Sum=0., Mean=None, Std=None, P95=None, P99=None, Maximum=None, Top1Mean=None)
    k = max(1, math.ceil(.01*n))
    return dict(Pixels=n, Sum=float(values.sum()), Mean=float(values.mean()), Std=float(values.std()),
        P95=float(np.quantile(values, .95)), P99=float(np.quantile(values, .99)), Maximum=float(values.max()),
        Top1Mean=float(np.partition(values, n-k)[-k:].mean()))


def measure(image, category, label, mask, stages, shot, thresholds=THRESHOLDS):
    """Raw/evaluated maps [H,W]. Return region, gap, paired attribution rows.

    Thresholds are predeclared diagnostic score cuts, NOT optimized operating
    points or calibrated probabilities. Paired effects compare final maps at
    the same stage/GT pixel; they do not prove which training mechanism caused it.
    """
    import numpy as np
    mask = np.asarray(mask, dtype=bool)
    selections = regions(mask, label)
    result, gaps, effects = [], [], []
    identity = dict(Shot=shot, Category=category, Image=image, Label=label)
    for stage, maps in stages.items():
        if set(maps) != set(CONDITIONS) or any(np.asarray(v).shape != mask.shape for v in maps.values()):
            raise ValueError('Need four same-sized maps and a matching GT mask.')
        for condition, values in maps.items():
            values = np.asarray(values)
            describe(values)  # Validate the full image, including empty GT regions.
            for region, selected in selections.items():
                data = values[selected]; stats = describe(data)
                extra = {}
                for t in thresholds:
                    high = int((data > t).sum())
                    extra.update({f'HighCount_{t}': high, f'HighRate_{t}': high/len(data) if len(data) else None,
                        f'MissRate_{t}': float((data <= t).mean()) if len(data) and region == 'defect' else None})
                result.append(dict(identity, Stage=stage, Condition=condition, Region=region, **stats, **extra))
            if label == 1:
                fg, bg = values[mask], values[~mask]
                gaps.append(dict(identity, Stage=stage, Condition=condition, DefectPixels=len(fg), BackgroundPixels=len(bg),
                    ForegroundMinusBackground=float(fg.mean()-bg.mean()) if len(fg) and len(bg) else None,
                    ForegroundMeanMinusBackgroundP99=float(fg.mean()-np.quantile(bg, .99)) if len(fg) and len(bg) else None))
        for pair, after, before in PAIRS:
            for region, selected in selections.items():
                a = np.asarray(maps[after])[selected]; b = np.asarray(maps[before])[selected]; n = len(a)
                for threshold in thresholds:
                    new_high = int(((a > threshold) & (b <= threshold)).sum())
                    removed_high = int(((a <= threshold) & (b > threshold)).sum())
                    effects.append(dict(identity, Stage=stage, Effect=pair, After=after, Before=before, Region=region,
                        Threshold=threshold, Pixels=n, DeltaSum=float((a.astype('float64')-b).sum()),
                        MeanDelta=float((a.astype('float64')-b).mean()) if n else None,
                        NewHighPixels=new_high, RemovedHighPixels=removed_high,
                        NewMissPixels=removed_high if region == 'defect' else None,
                        RecoveredPixels=new_high if region == 'defect' else None))
    return result, gaps, effects


def summaries(output, region_rows, effects, gaps):
    """Per-category and all-image summaries without caching full-test arrays.

    P95/P99/top1 are equal-image averages of WITHIN-image statistics, not pooled
    pixel quantiles. PooledMean/rates weight by pixel count. All_images is not
    a category macro average; per-category rows are always retained.
    """
    grouped = defaultdict(list); paired = defaultdict(list); margins = defaultdict(list)
    for row in region_rows:
        for category in (row['Category'], 'All_images'):
            grouped[category, row['Stage'], row['Condition'], row['Region']].append(row)
    output_rows = []
    for (category, stage, condition, region), rows in grouped.items():
        valid = [r for r in rows if r['Pixels']]; n = sum(r['Pixels'] for r in valid)
        one = dict(Category=category, Stage=stage, Condition=condition, Region=region, Images=len(rows),
            ValidImages=len(valid), Pixels=n, PooledMean=sum(r['Sum'] for r in valid)/n if n else None)
        for field in ('Mean', 'P95', 'P99', 'Maximum', 'Top1Mean'):
            one['AverageImage'+field] = sum(r[field] for r in valid)/len(valid) if valid else None
        for t in THRESHOLDS:
            high = sum(r[f'HighCount_{t}'] for r in valid)
            one[f'PooledHighRate_{t}'] = high/n if n else None
            one[f'PooledMissRate_{t}'] = (n-high)/n if n and region == 'defect' else None
        output_rows.append(one)
    for row in effects:
        for category in (row['Category'], 'All_images'):
            paired[category, row['Stage'], row['Effect'], row['Region'], row['Threshold']].append(row)
    pair_rows = []
    for (category, stage, effect, region, t), rows in paired.items():
        n = sum(r['Pixels'] for r in rows)
        new = sum(r['NewHighPixels'] for r in rows); removed = sum(r['RemovedHighPixels'] for r in rows)
        pair_rows.append(dict(Category=category, Stage=stage, Effect=effect, Region=region, Threshold=t, Pixels=n,
            PooledMeanDelta=sum(r['DeltaSum'] for r in rows)/n if n else None,
            NewHighRate=new/n if n else None, RemovedHighRate=removed/n if n else None,
            NetErrorRateChange=((removed-new) if region == 'defect' else (new-removed))/n if n else None))
    for row in gaps:
        for category in (row['Category'], 'All_images'):
            margins[category, row['Stage'], row['Condition']].append(row)
    gap_rows = []
    for (category, stage, condition), rows in margins.items():
        one = dict(Category=category, Stage=stage, Condition=condition, Images=len(rows))
        for field in ('ForegroundMinusBackground', 'ForegroundMeanMinusBackgroundP99'):
            values = [r[field] for r in rows if r[field] is not None]
            one['AverageImage'+field] = sum(values)/len(values) if values else None
            one['ValidImages'+field] = len(values)
        gap_rows.append(one)
    for name, rows in [('region_summary.csv', output_rows), ('paired_error_summary.csv', pair_rows), ('gap_summary.csv', gap_rows)]:
        if rows:
            write_csv(output/name, list(rows[0]), rows)
    tails = sorted((r for r in region_rows if r['Stage'] == 'evaluated' and r['Region'] != 'defect' and r['Pixels']),
                   key=lambda r: r['P99'], reverse=True)
    weak = sorted((r for r in gaps if r['Stage'] == 'evaluated' and r['ForegroundMeanMinusBackgroundP99'] is not None),
                  key=lambda r: r['ForegroundMeanMinusBackgroundP99'])
    for name, rows in [('background_tail_ranking.csv', tails), ('weak_defect_ranking.csv', weak)]:
        if rows:
            write_csv(output/name, list(rows[0]), rows)
