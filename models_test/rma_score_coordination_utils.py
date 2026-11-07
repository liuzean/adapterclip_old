"""Experiment H: source-only coefficient selection and original image metrics."""
import math
import statistics

from fusion_ablation_utils import write_csv

LAMBDAS = (0., .25, .5, .75, 1.)
IMAGE_FIELDS = ('I-AUROC', 'I-AP', 'I-F1max')
SELECTION = 'Equal source category-macro image-three-metric mean, then equal 1/2/4-shot mean; smaller lambda on 1e-9 ties.'


def mix_scores(existing, native, coefficient):
    """[N] float32 recorded probabilities; endpoints alias input exactly."""
    import torch
    if coefficient not in LAMBDAS or existing.shape != native.shape or existing.ndim != 1:
        raise ValueError('Need a predeclared coefficient and two same-shaped score vectors.')
    if any(not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any() for value in (existing, native)):
        raise ValueError('Nonfinite/out-of-bounds score.')
    if coefficient == 0:
        return existing
    if coefficient == 1:
        return native
    return (1-coefficient)*existing+coefficient*native


def image_metrics(rows, field, condition):
    """Use the unchanged repo Evaluator including trapezoidal PR-area AP."""
    import torch
    from branch_diagnostics_utils import evaluate_scores
    from tools import Evaluator
    result = []
    categories = list(dict.fromkeys(row['Category'] for row in rows))
    for category in categories:
        group = [row for row in rows if row['Category'] == category]
        labels = torch.tensor([int(row['Label']) for row in group]).int()
        scores = torch.tensor([float(row[field]) for row in group], dtype=torch.float32)
        dummy = torch.zeros((len(group), 1, 1))
        evaluator = Evaluator('cpu', metrics=list(IMAGE_FIELDS))
        values = evaluate_scores(evaluator, category, dummy, dummy, labels, scores, 'cpu', 'image')
        values = {'I-'+key: value for key, value in values.items()}
        if any(value is None or not math.isfinite(value) for value in values.values()):
            raise ValueError('Undefined source/target image metric.')
        result.append(dict(Category=category, Condition=condition, **values))
    result.append(dict(Category='Mean', Condition=condition,
                       **{field: statistics.mean(row[field] for row in result) for field in IMAGE_FIELDS}))
    return result


def source_selection(tables):
    """Only source tables accepted by caller; no target result is consulted."""
    if set(tables) != {1, 2, 4}:
        raise ValueError('All three source shots required.')
    candidates = []
    for coefficient in LAMBDAS:
        values = []
        for shot in (1, 2, 4):
            rows = [row for row in tables[shot] if row['Category'] == 'Mean' and float(row['Lambda']) == coefficient]
            if len(rows) != 1:
                raise ValueError('Missing/duplicate source candidate mean.')
            values.extend(float(rows[0][field]) for field in IMAGE_FIELDS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite source selection criterion.')
        candidates.append(dict(Lambda=coefficient, SourceScore=statistics.mean(values)))
    best = candidates[0]
    for row in candidates[1:]:
        if row['SourceScore'] > best['SourceScore']+1e-9:
            best = row
    return best['Lambda'], candidates


def save(path, rows):
    if rows:
        write_csv(path, list(rows[0]), rows)
