"""K supplementary dual output: exact K final image score + exact H final map."""
import math
from rma_utils import FIELDS, MetricsBuffer
from rma_align_ablation_utils import metric_index

DUAL = 'K_dual'
CONDITIONS = ('H', 'K_CRA', DUAL)
CRA_SHA = '7a0ea7beb98afc610ffa5390309d5ce3b1edfe0df37ab0cdd9e51ce750c8e150'


def route(maps, scores):
    """Alias complete final tensors; no fusion, normalization, max or rescoring."""
    maps[DUAL] = maps['H']
    scores[DUAL] = scores['K_CRA']


class DualBuffer(MetricsBuffer):
    """Two unique pixel buffers; dual metrics reuse the identical donor tensors."""
    def __init__(self, category):
        super().__init__(category, conditions=('H', 'K_CRA'))

    def append(self, mask, label, maps, scores, index):
        import torch
        if (not torch.equal(maps[DUAL][index], maps['H'][index])
                or not torch.equal(scores[DUAL][index], scores['K_CRA'][index])):
            raise ValueError('Dual output deviates from its complete H/K donors.')
        super().append(mask, label, maps, scores, index)

    def finish(self, device):
        pixels, images = super().finish(device)
        pixels.append(dict(next(r for r in pixels if r['Condition'] == 'H'), Condition=DUAL))
        images.append(dict(next(r for r in images if r['Condition'] == 'K_CRA'), Condition=DUAL))
        return pixels, images


def compare(current, history, baseline, classes, offline):
    """Check full H/K history, donor identity, offline means and ALL differences."""
    fresh = metric_index(current, CONDITIONS, classes)
    old = metric_index(history, ('H', 'K_CRA'), classes)
    a = metric_index(baseline, ('baseline',), classes)
    checks, identity, deltas = [], [], []
    for category in (*classes, 'Mean'):
        for condition in ('H', 'K_CRA'):
            for f in FIELDS:
                value = float(fresh[category, condition][f])-float(old[category, condition][f])
                checks.append(dict(Category=category, Condition=condition, Metric=f, DeltaPP=value,
                                   Passed=math.isfinite(value) and abs(value) <= .001))
        dual = fresh[category, DUAL]
        for f in FIELDS:
            donor = 'K_CRA' if f.startswith('I-') else 'H'
            value = float(dual[f])-float(fresh[category, donor][f])
            identity.append(dict(Category=category, Metric=f, Donor=donor, DeltaPP=value, Passed=value == 0))
        for name, row in (('H', fresh[category, 'H']), ('K_CRA', fresh[category, 'K_CRA']), ('baseline', a[category, 'baseline'])):
            change = {f:float(dual[f])-float(row[f]) for f in FIELDS}
            deltas.append(dict(Category=category, Condition=DUAL, Comparator=name, **change,
                ImageNet=sum(change[f] for f in FIELDS[:3]), PixelNet=sum(change[f] for f in FIELDS[3:]), Net=sum(change.values())))
    offline_checks = []
    for f in FIELDS:
        value = float(fresh['Mean', DUAL][f])-float(offline[f])
        offline_checks.append(dict(Metric=f, DeltaPP=value, Passed=math.isfinite(value) and abs(value) <= .001))
    return checks, identity, offline_checks, deltas
