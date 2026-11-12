"""Experiment M: frozen PQA features, shared 130-parameter replacement head.

PQAdapter.forward remains the authoritative feature/matching implementation.
Only branch aggregation is exposed here to preserve gradients through the new
classifier. Runtime checks compare it with the existing infer_batch path.
"""
import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def make_classifier(pq):
    """Copy one shared Conv2d(64,2,1,bias=True); never modify the old head."""
    original = pq.local_adapter[0][-1]
    if (not isinstance(original, torch.nn.Conv2d) or original.in_channels != 64
            or original.out_channels != 2 or original.kernel_size != (1, 1) or original.bias is None
            or any(layer is not pq.local_adapter[0] or layer[-1] is not original for layer in pq.local_adapter)):
        raise ValueError('Requires the original four-layer shared 64->2 classifier.')
    head = copy.deepcopy(original).requires_grad_(True).eval()
    if sum(p.numel() for p in head.parameters()) != 130:
        raise ValueError('Classifier budget must be 130 distinct trainable parameters.')
    return head


class FeatureTap:
    """Capture four frozen [B,64,h,w] inputs without replacing any old output."""
    def __init__(self, pq):
        self.values = None
        self.handle = pq.local_adapter[0][-1].register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if self.values is not None: self.values.append(args[0].detach())

    @contextmanager
    def capture(self):
        if self.values is not None: raise RuntimeError('Nested classifier capture.')
        self.values = []
        try:
            yield self.values
            if len(self.values) != 4: raise ValueError('Expected four uses of one shared classifier.')
        finally:
            self.values = None

    def close(self):
        self.handle.remove()


def classify(head, features, size):
    """Four frozen features -> mean anomaly probability [B,H,W], differentiable."""
    if len(features) != 4: raise ValueError('Missing PQA layers.')
    layers = [F.interpolate(head(x).softmax(1), (size, size), mode='bilinear')[:, 1] for x in features]
    return torch.concat([v.unsqueeze(1) for v in layers], dim=1).mean(dim=1)


def local_raw(va, ta, pqa, rma, align):
    """Exactly the fixed 12.5% mixture followed by unchanged ALIGN harmonic fusion."""
    from adaptcliplib import fusion_fun
    old_mean = fusion_fun([va, ta, pqa], fusion_type='average_mean')
    return fusion_fun([.875*old_mean + .125*rma, align], fusion_type='harmonic_mean')


class FrozenPaths:
    """Old K scores and frozen features are computed without any adapted head."""
    def __init__(self, state, cra_hook):
        self.state, self.hook = state, cra_hook
        self.tap = FeatureTap(state.adapters[2])

    def extract(self, features, patches, pf, pp, *, arms=('plain', 'cra'), verify=False, paths=()):
        from adaptcliplib import fusion_fun
        from branch_diagnostics_utils import smooth_maps
        from fusion_ablation import infer_batch
        from h_inference import select_score
        from rma_output_ablation_utils import route_outputs
        state = self.state
        textual, visual, pq = state.adapters
        frozen = (features, pf, *patches, *pp)
        versions = [v._version for v in frozen]
        branches = {}
        with torch.no_grad():
            va_g, va_l = visual(features, patches, textual.static_text_features)
            ta_g, ta_l = textual.compute_global_local_score(features, patches, state.text)
            rma_g, rma_l = state.rma(patches, pp)
            rma = rma_l.softmax(1)[:, 1]
            for arm in arms:
                before = self.hook.calls
                with self.hook.active(arm == 'cra'), self.tap.capture() as captured:
                    pq_g, pq_l, align_layers = pq(features, patches, pf, pp)
                if self.hook.calls-before != (4 if arm == 'cra' else 0):
                    raise ValueError('CRA must run at exactly the four intended PQA layers.')
                pqa = torch.concat([v[:, 1].unsqueeze(1) for v in pq_l], dim=1).mean(dim=1)
                align = fusion_fun(align_layers, fusion_type='harmonic_mean')[:, 0]
                pq_score = torch.concat([v.softmax(-1).unsqueeze(-1) for v in pq_g], dim=-1).mean(-1)[:, 1]
                native = fusion_fun([va_g.softmax(-1)[:, 1], ta_g.softmax(-1)[:, 1], pq_score], fusion_type='average_mean')
                mean = fusion_fun([va_l[:, 1], ta_l[:, 1], pqa], fusion_type='average_mean')
                raw = local_raw(va_l[:, 1], ta_l[:, 1], pqa, rma, align)
                branches[arm] = dict(features=list(captured), pqa=pqa, align=align, native=native,
                                     raw=raw, va=va_l[:, 1], ta=ta_l[:, 1], rma=rma)
                if arm == 'cra':
                    # Image donor uses ORIGINAL classifier and 25% local-score route.
                    raw25 = fusion_fun([(3*mean+rma)/4, align], fusion_type='harmonic_mean')
                    map25 = smooth_maps(raw25, state.config.sigma).cpu()
                    se = fusion_fun([native, map25.flatten(1).max(1).values.to(native.device)], fusion_type='harmonic_mean').cpu()
                    branches[arm]['image_score'] = select_score(se, native.cpu(), 'H')
                if verify:
                    with self.hook.active(arm == 'cra'):
                        old, maps, scores, _, _, n, extra = infer_batch(
                            state.config, features, patches, pf, pp, *state.adapters, state.text, paths)
                    if not torch.equal(mean, old['no_align']) or not torch.equal(align, extra['ALIGN']):
                        raise ValueError('M branch aggregation differs from existing inference.')
                    expected = fusion_fun([.875*old['no_align']+.125*rma, extra['ALIGN']], fusion_type='harmonic_mean')
                    if not torch.equal(raw, expected): raise ValueError('M localization differs from H/K fixed12.5 path.')
                    if arm == 'cra':
                        _, _, scored, _ = route_outputs(old, maps, scores, n, extra, rma_g, rma_l, state.config.sigma)
                        score = select_score(scored['G0_L1'], n.cpu(), 'H')
                        if not torch.equal(score, branches[arm]['image_score']):
                            raise ValueError('M frozen image donor differs from existing K image route.')
            if len(branches) == 2 and not torch.equal(branches['plain']['align'], branches['cra']['align']):
                raise ValueError('CRA altered reference ALIGN.')
        if versions != [v._version for v in frozen]: raise ValueError('Frozen shared features were modified.')
        return branches

    def batch(self, images, classes, *, verify=False, paths=()):
        from test import prompt_association
        state = self.state
        with torch.no_grad():
            features, patches = state.model.encode_image(images.to(state.device), state.config.features_list, DPAM_layer=20)
            pf, pp = prompt_association(*state.memory, classes)
            pf, pp = pf.to(state.device), [v.to(state.device) for v in pp]
        return self.extract(features, patches, pf, pp, verify=verify, paths=paths)

    def adapted_raw(self, branch, classifier):
        pqa = classify(classifier, branch['features'], self.state.config.image_size)
        return local_raw(branch['va'], branch['ta'], pqa, branch['rma'], branch['align'])

    def close(self):
        self.tap.close()


def evaluate(state, engine, heads, loader, output, expected, *, epoch0=False, history_scores=None, progress=None):
    """Full metrics with one-category buffers; all conditions alias the same K score."""
    from branch_diagnostics_utils import smooth_maps
    from fusion_ablation_utils import path_key, write_csv
    from rma_utils import MetricsBuffer, cleanup, save_tables
    conditions = ('K_dual', 'K_original_local', 'M_plain', 'M_cra') if len(heads) == 2 else ('unadapted', 'adapted')
    pixels, images, score_rows = [], [], []
    buffer, count, closed = None, 0, set()
    def flush():
        p, i = buffer.finish(state.device); pixels.extend(p); images.extend(i)
        save_tables(output, pixels, images); cleanup()
    with torch.no_grad():
        for step, items in enumerate(loader, 1):
            if progress: progress(step, count)
            branches = engine.batch(items['img'], items['cls_name'], verify=step == 1, paths=items['img_path'])
            score = branches['cra']['image_score']
            maps = {}
            if len(heads) == 2:
                maps['K_dual'] = smooth_maps(branches['plain']['raw'], state.config.sigma).cpu()
                maps['K_original_local'] = smooth_maps(branches['cra']['raw'], state.config.sigma).cpu()
                for arm, head in heads.items():
                    maps['M_'+arm] = smooth_maps(engine.adapted_raw(branches[arm], head), state.config.sigma).cpu()
            else:
                arm, head = next(iter(heads.items()))
                raw = engine.adapted_raw(branches[arm], head)
                if epoch0 and not torch.equal(raw, branches[arm]['raw']):
                    raise ValueError('Epoch0 copied head does not restore original plain/CRA localization.')
                maps['unadapted'] = smooth_maps(branches[arm]['raw'], state.config.sigma).cpu()
                maps['adapted'] = smooth_maps(raw, state.config.sigma).cpu()
            if any(v.shape != items['img_mask'][:, 0].shape or not torch.isfinite(v).all() for v in maps.values()):
                raise ValueError('Invalid localization shape or values.')
            if not torch.isfinite(score).all(): raise ValueError('Invalid frozen image scores.')
            scores = {n:score for n in conditions}
            for i, category in enumerate(items['cls_name']):
                if count >= len(expected): raise ValueError('Extra evaluation query.')
                row, path, label = expected[count], items['img_path'][i], int(items['anomaly'][i])
                if (category != row['Category'] or path_key(path) != path_key(row['Image Path'])
                        or label != int(row['Sample Type'] == 'Anomaly')):
                    raise ValueError('Query order, identity or label differs from locked manifest.')
                if history_scores is not None and abs(float(score[i])-float(history_scores[count]['K_dual'])) > 1e-6:
                    raise ValueError('Actual fixed K image score differs from historical K-dual.')
                if buffer is None or category != buffer.category:
                    if buffer is not None: flush(); closed.add(buffer.category)
                    if category in closed: raise ValueError('Category queries must be contiguous.')
                    buffer = MetricsBuffer(category, conditions=conditions)
                buffer.append(items['img_mask'][i, 0] > .5, label, maps, scores, i)
                score_rows.append(dict(Category=category, **{'Image Path':path}, Label=label, K_dual=float(score[i])))
                count += 1
            del branches, maps, scores
    if buffer is None or count != len(expected): raise ValueError('Incomplete full-set evaluation.')
    flush()
    table = save_tables(output, pixels, images)
    write_csv(output/'image_scores.csv', list(score_rows[0]), score_rows)
    return table
