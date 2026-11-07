"""H inference routing and parity checks; scheduler-safe imports."""
import math

from fusion_ablation_utils import path_key, read_csv, write_csv
from rma_reliability_utils import ReliabilityBuffer
from rma_score_coordination_utils import mix_scores
from rma_utils import FIELDS, digest

LOCAL = 'local_candidate'
CONDITIONS = ('baseline', 'G0_L1', 'G1_L1', LOCAL, 'E', 'H')


class HRouter:
    """Capture actual G0 from the existing fusion call, never invert a score."""
    def __init__(self, historical):
        self.historical = historical
        self.native = None
        self.rows = []

    def fuse(self, old, fixed, extra, logits, dispersion, fraction, sigma, endpoint):
        from rma_fixed_fusion_utils import gated_output
        self.native = fixed.detach().cpu()
        return gated_output(old, fixed, extra, logits, dispersion, fraction, sigma, endpoint)

    def route(self, raw, maps, scores):
        import torch
        if self.native is None or self.native.shape != scores['G0_L1'].shape:
            raise ValueError('Actual native G0 was not captured for this batch.')
        raw['E'] = raw[LOCAL]; maps['E'] = maps[LOCAL]; scores['E'] = scores['G0_L1']
        raw['H'] = raw['E']; maps['H'] = maps['E']
        scores['H'] = mix_scores(scores['E'], self.native, .25)
        if not torch.equal(maps['H'], maps['E']) or not torch.equal(raw['H'], raw['E']):
            raise ValueError('H pixel path diverged from freshly computed E.')
        for i in range(len(self.native)):
            previous = self.historical[len(self.rows)]
            values = dict(G0=float(self.native[i]), E=float(scores['E'][i]), H=float(scores['H'][i]))
            errors = {name: values[name]-float(previous[name]) for name in values}
            if not all(math.isfinite(delta) and abs(delta) <= 1e-6 for delta in errors.values()):
                raise ValueError(f'Actual native/E/H score disagrees with offline H: {errors}')
            self.rows.append(dict(Category=previous['Category'], **{'Image Path': previous['Image Path']},
                                  Label=int(previous['Label']), **values, G0Delta=errors['G0'], EDelta=errors['E'],
                                  HDelta=errors['H'], RawPixelEqual=True, FinalPixelEqual=True))
        self.native = None


class HBuffer(ReliabilityBuffer):
    """Only three unique full pixel buffers; H/E alias the evaluated local map."""
    def __init__(self, category, conditions=CONDITIONS):
        super().__init__(category, conditions=('baseline', 'G0_L1', 'G1_L1', LOCAL))
        self.h_scores = []

    def append(self, mask, label, maps, scores, index):
        import torch
        if not torch.equal(maps['H'][index], maps['E'][index]) or not torch.equal(maps['E'][index], maps[LOCAL][index]):
            raise ValueError('H/E must preserve the same complete final pixel map.')
        if not torch.equal(scores['E'][index], scores['G0_L1'][index]):
            raise ValueError('E complete final image score changed.')
        super().append(mask, label, maps, scores, index)
        self.h_scores.append(scores['H'][index].detach().cpu().clone())

    def finish(self, device):
        import torch
        from branch_diagnostics_utils import evaluate_scores
        from tools import Evaluator
        labels = torch.tensor(self.labels).int()
        pixels, images = super().finish(device)
        for name in ('E', 'H'):
            pixels.append(dict(next(row for row in pixels if row['Condition'] == LOCAL), Condition=name))
        images.append(dict(next(row for row in images if row['Condition'] == 'G0_L1'), Condition='E'))
        dummy = torch.zeros((len(labels), 1, 1))
        evaluator = Evaluator(device, metrics=['I-AUROC', 'I-AP', 'I-F1max'])
        images.append(dict(Category=self.category, Condition='H', **evaluate_scores(
            evaluator, self.category, dummy, dummy, labels, torch.stack(self.h_scores), device, 'image')))
        self.h_scores.clear()
        return pixels, images


def final_check(output, table, router, historical_dir, e_dir):
    """Whole-query score parity, fresh E six metrics, H offline six metrics."""
    fresh = read_csv(output/'image_scores.csv')
    if len(fresh) != len(router.rows) or len(fresh) != len(router.historical):
        raise ValueError('Incomplete query coverage.')
    for actual, checked in zip(fresh, router.rows):
        if (actual['Category'] != checked['Category'] or path_key(actual['Image Path']) != path_key(checked['Image Path'])
                or int(actual['Label']) != checked['Label']):
            raise ValueError('H offline query identity/order/label mismatch.')
    write_csv(output/'H_score_and_pixel_checks.csv', list(router.rows[0]), router.rows)
    current = {(row['Category'], row['Condition']): row for row in table}
    old_h = {(row['Category'], row['Condition']): row for row in read_csv(historical_dir/'metrics_combined.csv')}
    old_e = {(row['Category'], row['Condition']): row for row in read_csv(e_dir/'metrics_combined.csv')}
    categories = {category for category, name in current if name == 'H'}
    if categories != {category for category, name in old_h if name == 'H'}:
        raise ValueError('Offline category coverage differs.')
    checks = []
    for category in categories:
        for name, previous in (('H', old_h[category, 'H']), ('E', old_e[category, 'dual_output'])):
            for field in FIELDS:
                delta = float(current[category, name][field])-float(previous[field])
                checks.append(dict(Category=category, Condition=name, Metric=field, DeltaPP=delta,
                                   Passed=math.isfinite(delta) and abs(delta) <= .001))
        for field in FIELDS[3:]:
            if float(current[category, 'H'][field]) != float(current[category, 'E'][field]):
                raise ValueError('Fresh H and E pixel metrics must be exactly equal.')
    write_csv(output/'H_offline_metric_checks.csv', list(checks[0]), checks)
    if not all(row['Passed'] for row in checks):
        raise ValueError('H offline or historical E six metrics differ.')
    names = ('image_scores.csv', 'metrics_combined.csv', 'H_score_and_pixel_checks.csv', 'H_offline_metric_checks.csv',
             'references.csv', 'sample_manifest.csv', 'input_audit.json')
    return dict(H_offline_score_parity_passed=True, H_pixel_identity_passed=True, H_six_metric_parity_passed=True,
                H_coefficient=.25, H_full_pixel_evaluation=True,
                H_output_sha256={name: digest(output/name) for name in names})
