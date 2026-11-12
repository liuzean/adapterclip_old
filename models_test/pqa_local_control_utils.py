"""Experiment M contracts. Schedulers import only standard-library modules."""
import math
from pathlib import Path

from cra_dual_output_utils import CRA_SHA
from cra_utils import checked_checkpoint, code_hashes as base_hashes, source_plan
from fusion_ablation_utils import path_key, read_csv, read_json, write_csv
from h_inference import WEIGHT_SHA, recorded_inputs
from rma_align_ablation_utils import DATASETS, metric_index, verified_history
from rma_control_io import write_json
from rma_utils import FIELDS, digest

ROOT = Path(__file__).resolve().parents[1]
H_WEIGHT = ROOT/'results/train_rma_scratch_new/mvtec_seed10/best.pth'
CRA_WEIGHT = ROOT/'results/train_cra/mvtec_seed10/best.pth'
REFERENCE_RUN = ROOT/'results/rma_score_coordination_inference/experiment_H_inference_seed10'
DUAL_RUN = ROOT/'results/cra_dual_output/K_dual_seed10'
FORMAT = 'adaptclip-pqa-local-130-v1'
CONDITIONS = ('K_dual', 'K_original_local', 'M_plain', 'M_cra')
ARMS = ('plain', 'cra')
PIXEL_FIELDS = FIELDS[3:]
SETTINGS = dict(seed=10, epochs=15, batch_size=2, accumulation=4, optimizer='adamw',
                learning_rate=.0001, betas=[.9, .999], weight_decay=.0001, gradient_clip=1.,
                num_workers=0, train_reference_shots=[1, 2, 4], trainable_parameters=130,
                train_local_fraction=.125, deploy_local_fraction=.125, sigma=4,
                loss='final pre-Gaussian ALIGN map: Focal + anomaly Dice + normal Dice',
                selection='source pixel-three-metric category macro, then mean over 1/2/4-shot; earliest exact maximum including epoch0',
                image_score='frozen original K image path, original local head, .75*SE25+.25*G0')
NEW_FILES = ('pqa_local_control_utils.py', 'pqa_local_control_model.py',
             'train_pqa_local_control.py', 'test_pqa_local_control.py', 'pqa_local_control_pipeline.py')


def code_hashes():
    return dict(base_hashes(), **{str(ROOT/'models_test'/n): digest(ROOT/'models_test'/n) for n in NEW_FILES})


def verify_plan(plan):
    if plan['code_sha256'] != code_hashes():
        raise ValueError('Experiment M implementation changed; do not mix campaigns.')
    for name, sha in plan['input_sha256'].items():
        if digest(name) != sha:
            raise ValueError('Experiment M input changed: '+name)


def training_plan():
    source, split, grouped, old = source_plan(H_WEIGHT)
    _, status, sha = checked_checkpoint(CRA_WEIGHT)
    if sha != CRA_SHA or status['best_epoch'] != 9:
        raise ValueError('Requires fixed CRA epoch9, not a newly selected CRA.')
    paths = [CRA_WEIGHT, *(CRA_WEIGHT.parent/n for n in
             ('campaign_config.json','run_status.json','training_history.csv','source_H_metrics.csv'))]
    plan = dict(old, experiment='M', code_sha256=code_hashes(), settings=SETTINGS,
                cra_checkpoint=str(CRA_WEIGHT), cra_checkpoint_sha256=CRA_SHA,
                trainable='copy of shared pq.local_adapter[0][-1], 64->2 biased 1x1 convolution',
                source_selection_uses_target_labels=False)
    plan['input_sha256'] = dict(old['input_sha256'], **{str(p):digest(p) for p in paths})
    return source, split, grouped, plan


def select_source(rows):
    selected = [r for r in rows if r['Category'] == 'Mean' and r['Condition'] == 'adapted']
    if sorted(int(r['Shot']) for r in selected) != [1, 2, 4]:
        raise ValueError('Source validation must contain all three shots.')
    values = [float(r[f]) for r in selected for f in PIXEL_FIELDS]
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in values):
        raise ValueError('Invalid source pixel metrics.')
    return sum(values)/9


def checked_training(directory, arm):
    directory = Path(directory).resolve()
    directory.relative_to((ROOT/'results/train_pqa_local_control').resolve())
    plan, status = read_json(directory/'campaign_config.json'), read_json(directory/'run_status.json')
    verify_plan(plan)
    if plan['arm'] != arm or plan['settings'] != SETTINGS:
        raise ValueError('Wrong local-control arm or settings.')
    flags = ('complete', 'frozen_integrity_passed', 'gradient_checks_passed', 'epoch0_identity_passed')
    if not all(status.get(k) is True for k in flags) or status.get('completed_epochs') != 15:
        raise ValueError('Requires completed 15-epoch local adaptation with all checks passed.')
    history = read_csv(directory/'training_history.csv')
    if ([int(r['epoch']) for r in history] != list(range(16))
            or digest(directory/'training_history.csv') != status['training_history_sha256']):
        raise ValueError('Incomplete or edited source selection history.')
    if any(not math.isfinite(float(r['validation_score'])) for r in history):
        raise ValueError('Nonfinite selection score.')
    best = max(history, key=lambda r: float(r['validation_score']))
    weight = directory/'best.pth'
    if (int(best['epoch']) != status['best_epoch'] or float(best['validation_score']) != status['best_score']
            or digest(weight) != status['selected_checkpoint_sha256']):
        raise ValueError('Weight differs from source-selected earliest maximum.')
    return plan, status, dict(path=str(weight), sha256=digest(weight), epoch=status['best_epoch'],
                              arm=arm, campaign_sha256=digest(directory/'campaign_config.json'))


def target_inputs(reference_run, dual_run, dataset, shot):
    from cra_dual_output import verified_complete
    inputs = recorded_inputs(reference_run, dataset, shot)
    original, _ = verified_history(inputs)
    state = read_json(Path(dual_run)/'run_status.json')
    if state.get('complete') is not True:
        raise ValueError('Historical K-dual campaign is incomplete.')
    folder = Path(state['job_results'][f'{dataset}/{shot}']).resolve()
    folder.relative_to(Path(dual_run).resolve())
    verified_complete(folder, len(inputs['manifest']))
    cfg = read_json(folder/'experiment_config.json')
    if digest(cfg['h_checkpoint']) != WEIGHT_SHA or digest(cfg['cra_checkpoint']) != CRA_SHA:
        raise ValueError('Historical donors have different H/CRA weights.')
    for name in ('references.csv', 'sample_manifest.csv'):
        if read_csv(folder/name) != read_csv(inputs['history']/name):
            raise ValueError('Historical K-dual and H identities differ.')
    table = read_csv(folder/'metrics_combined.csv')
    metric_index(table, ('H', 'K_CRA', 'K_dual'), inputs['classes'])
    scores = read_csv(folder/'image_scores.csv')
    if len(scores) != len(inputs['manifest']):
        raise ValueError('Historical K score coverage differs.')
    for row, expected in zip(scores, inputs['manifest']):
        if (row['Category'] != expected['Category'] or path_key(row['Image Path']) != path_key(expected['Image Path'])
                or int(row['Label']) != int(expected['Sample Type'] == 'Anomaly')):
            raise ValueError('Historical K score identities differ.')
    return inputs, folder, table, scores, original


def target_plan(plain_run, cra_run, reference_run=REFERENCE_RUN, dual_run=DUAL_RUN):
    left, _, plain = checked_training(plain_run, 'plain')
    right, _, cra = checked_training(cra_run, 'cra')
    if ({k:v for k,v in left.items() if k != 'arm'} != {k:v for k,v in right.items() if k != 'arm'}):
        raise ValueError('Two arms must use identical data, settings, initialization and code.')
    paths = [Path(p)/n for p in (plain_run, cra_run) for n in
             ('campaign_config.json', 'run_status.json', 'best.pth', 'training_history.csv')]
    paths += [Path(dual_run)/'run_status.json']
    jobs = {}
    for dataset in DATASETS:
        for shot in (1, 2, 4):
            inputs, folder, _, _, _ = target_inputs(reference_run, dual_run, dataset, shot)
            paths += [folder/n for n in ('run_status.json', 'experiment_config.json', 'references.csv',
                                        'sample_manifest.csv', 'metrics_combined.csv', 'image_scores.csv')]
            paths += [inputs['history']/n for n in ('experiment_config.json', 'references.csv', 'sample_manifest.csv',
                       'run_status.json','input_audit.json','metrics_combined.csv','image_scores.csv')]
            paths += [inputs['meta']]
            jobs[f'{dataset}/{shot}'] = dict(queries=len(inputs['manifest']))
    plan = dict(left, experiment='M_target', weights=dict(plain=plain, cra=cra), jobs=jobs,
                reference_run=str(reference_run), dual_run=str(dual_run))
    plan.pop('arm')
    plan['input_sha256'] = dict(left['input_sha256'], **{str(p):digest(p) for p in paths})
    return plan


def compare(table, historical, original, classes):
    fresh = metric_index(table, CONDITIONS, classes)
    old = metric_index(historical, ('H', 'K_CRA', 'K_dual'), classes)
    a = metric_index(original, ('baseline', 'H'), classes)
    checks, deltas = [], []
    for category in (*classes, 'Mean'):
        for condition, donor in (('K_dual', 'K_dual'), ('K_original_local', 'K_CRA')):
            for f in FIELDS:
                delta = float(fresh[category, condition][f])-float(old[category, donor][f])
                checks.append(dict(Category=category, Condition=condition, Metric=f, DeltaPP=delta,
                                   Passed=math.isfinite(delta) and abs(delta) <= .001))
        for condition in CONDITIONS:
            for f in FIELDS[:3]:
                if float(fresh[category, condition][f]) != float(fresh[category, 'K_dual'][f]):
                    raise ValueError('Adapted localization affected fixed K image metrics.')
        for condition in ('M_plain', 'M_cra'):
            comparisons = [('K_dual', fresh[category, 'K_dual']), ('K_original_local', fresh[category, 'K_original_local']),
                           ('original_A', a[category, 'baseline'])]
            if condition == 'M_cra': comparisons.append(('M_plain', fresh[category, 'M_plain']))
            for name, reference in comparisons:
                change = {f:float(fresh[category, condition][f])-float(reference[f]) for f in FIELDS}
                deltas.append(dict(Category=category, Condition=condition, Comparator=name, **change,
                    ImageNet=sum(change[f] for f in FIELDS[:3]), PixelNet=sum(change[f] for f in FIELDS[3:]), Net=sum(change.values())))
    return checks, deltas


def summarize(session, state):
    metrics, differences = [], []
    for key, folder in state['job_results'].items():
        dataset, shot = key.split('/')
        for filename, collection in (('metrics_combined.csv', metrics), ('deltas.csv', differences)):
            collection.extend(dict(Dataset=dataset, Shot=int(shot), **r) for r in read_csv(Path(folder)/filename))
    for name, rows in (('metrics_summary.csv', [r for r in metrics if r['Category'] == 'Mean']),
                       ('metrics_by_category.csv', [r for r in metrics if r['Category'] != 'Mean']),
                       ('deltas_by_category.csv', differences)):
        write_csv(session/name, list(rows[0]), rows)
    records, overall = [], []
    for condition, comparator in sorted({(r['Condition'],r['Comparator']) for r in differences}):
        passed = 0
        for shot in (1, 2, 4):
            rows = [r for r in differences if r['Category'] == 'Mean' and r['Condition'] == condition
                    and r['Comparator'] == comparator and int(r['Shot']) == shot]
            if {r['Dataset'] for r in rows} != set(DATASETS) or len(rows) != 6:
                raise ValueError('Cannot judge retention on incomplete datasets.')
            values = [float(r['Net']) for r in rows]
            good, net = sum(v > 0 for v in values), sum(values)
            keep = good >= 3 and net > 0
            records.append(dict(Condition=condition, Comparator=comparator, Shot=shot,
                                PositiveDatasets=good, NetPP=net, Pass=keep))
            passed += int(keep)
        overall.append(dict(Condition=condition, Comparator=comparator, PassingShots=passed, Retain=passed >= 2))
    write_csv(session/'retention_by_shot.csv', list(records[0]), records)
    write_json(session/'retention_summary.json', dict(results=overall,
        note='CRA extra-value evidence requires M_cra versus equally trained M_plain, separately from improvement versus K_dual.'))
