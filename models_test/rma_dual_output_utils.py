"""Output routing and offline parity checks for the old-B dual-output experiment."""
import math

from fusion_ablation_utils import read_csv, read_json, write_csv
from rma_utils import FIELDS, MetricsBuffer

CONDITIONS = ('baseline', 'rma', 'dual_output')


def route_outputs(raw, maps, scores):
    """Alias B's complete raw/smoothed BxHxW maps and A's B-vector image score.

    No re-fusion, re-smoothing, normalization, or in-place tensor arithmetic.
    """
    raw['dual_output'] = raw['rma']
    maps['dual_output'] = maps['rma']
    scores['dual_output'] = scores['baseline']


class DualOutputBuffer(MetricsBuffer):
    """Evaluate A/B once and reuse their identical dual-output metric inputs.

    Only two sets of pixel arrays are buffered on CPU. A third full-map copy
    and duplicate evaluation are unnecessary: dual image inputs equal A,
    and dual pixel inputs equal B on every sample, checked before appending.
    """

    def append(self, mask, label, maps, scores, index):
        import torch
        if not torch.equal(maps['dual_output'][index], maps['rma'][index]):
            raise RuntimeError('Dual pixel map differs from the complete B final map.')
        if not torch.equal(scores['dual_output'][index], scores['baseline'][index]):
            raise RuntimeError('Dual image score differs from original A final score.')
        super().append(mask, label, maps, scores, index)

    def finish(self, device):
        pixels, images = super().finish(device)
        pixels.append(dict(next(r for r in pixels if r['Condition'] == 'rma'), Condition='dual_output'))
        images.append(dict(next(r for r in images if r['Condition'] == 'baseline'), Condition='dual_output'))
        return pixels, images


def check_history(output, historical):
    """Audit same query/reference identities, A/B six metrics and per-image scores.

    Metrics tolerate .001 percentage points; scores use the existing runtime
    tolerance (atol=1e-6, rtol=1e-5). A failure prevents round completion.
    """
    for name in ('references.csv', 'sample_manifest.csv'):
        if read_csv(output/name) != read_csv(historical/name):
            raise ValueError(f'Old-B identities changed: {name}')
    configs = [read_json(d/'experiment_config.json') for d in (output, historical)]
    for field in ('seed', 'k_shots', 'dataset', 'test_data_path', 'checkpoint_path', 'pretrained_model',
                  'batch_size', 'image_size', 'features_list', 'sigma', 'fusion_type', 'n_ctx',
                  'vl_reduction', 'pq_mid_dim', 'pq_context', 'visual_learner', 'textual_learner',
                  'pq_learner', 'rma_sha256', 'disable_rma'):
        if configs[0][field] != configs[1][field]:
            raise ValueError(f'Old-B protocol changed: {field}')
    tables = [{(r['Category'], r['Condition']): r for r in read_csv(d/'metrics_combined.csv')}
              for d in (output, historical)]
    current, old = tables
    if set(old) != {k for k in current if k[1] != 'dual_output'}:
        raise ValueError('Historical A/B category or condition coverage changed.')
    checks = []
    for key, previous in old.items():
        for field in FIELDS:
            delta = float(current[key][field])-float(previous[field])
            checks.append(dict(Category=key[0], Condition=key[1], Metric=field,
                               Delta=delta, Passed=math.isfinite(delta) and abs(delta) <= .001))
    write_csv(output/'old_B_metric_checks.csv', list(checks[0]), checks)
    if not all(r['Passed'] for r in checks):
        raise RuntimeError('Historical A/B metric parity failed; see old_B_metric_checks.csv.')
    left, right = [read_csv(d/'image_scores.csv') for d in (output, historical)]
    if len(left) != len(right):
        raise ValueError('Historical per-image score coverage changed.')
    score_checks = []
    for a, b in zip(left, right):
        same = all(a[f] == b[f] for f in ('Category', 'Image Path', 'Label'))
        same = same and all(math.isclose(float(a[f]), float(b[f]), abs_tol=1e-6, rel_tol=1e-5)
                            for f in ('baseline Image Score', 'rma Image Score'))
        score_checks.append(dict(Image=a['Image Path'], Passed=same))
        if float(a['dual_output Image Score']) != float(a['baseline Image Score']):
            raise RuntimeError('Saved dual image score differs from A.')
    write_csv(output/'old_B_score_checks.csv', list(score_checks[0]), score_checks)
    if not all(r['Passed'] for r in score_checks):
        raise RuntimeError('Historical A/B per-image score parity failed.')
    for category in {key[0] for key in old}:
        for field in FIELDS:
            source = 'baseline' if field.startswith('I-') else 'rma'
            if current[category, 'dual_output'][field] != current[category, source][field]:
                raise RuntimeError('Dual metric routing mismatch.')
    return dict(old_B_parity_passed=True, dual_output_parity_passed=True,
                historical_metric_checks=len(checks), historical_score_checks=len(score_checks))
