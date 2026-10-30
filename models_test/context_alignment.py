"""Four ALIGN conditions from one encoder/adapter pass; frozen original model.

Root: python models_test/context_alignment.py
Default: VisA/BTAD/MPDD/SDD/DTD/part_glass, 1/2/4-shot, seed=10.
Sequential dataset -> shot workers; --shots can restrict the shot counts.
"""
import argparse
import gc
import hashlib
import logging
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from branch_ablation_utils import select_samples
from clip_subspace_utils import audit_controls, check_reference_separation
from context_alignment_utils import (
    CONDITIONS, PAIRS, CategoryBuffer, ContextMatcher, RawTap, initialize_both,
)
from dual_output_utils import FIELDS
from fusion_ablation import infer_batch
from fusion_ablation_utils import (
    audit_source, compare_baseline, metric_tables, new_directory, path_key,
    project_path, read_csv, read_json, save_panels, write_csv, write_json,
)

DATASETS = ('Visa', 'BTAD', 'MPDD', 'SDD', 'DTD', 'part_glass')
IDS = dict(zip(DATASETS, ('visa', 'btad', 'mpdd', 'SDD', 'DTD', 'part_glass')))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--match_chunk', type=int, default=256, help='Query-token chunk; no reference subsampling.')
    p.add_argument('--disable_context_alignment', action='store_true', help='All candidates alias A for recovery.')
    p.add_argument('--save_maps', action='store_true')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--_session', help=argparse.SUPPRESS)
    p.add_argument('--_dataset', choices=DATASETS, help=argparse.SUPPRESS)
    p.add_argument('--_shot', type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_workers < 0 or args.match_chunk < 1:
        p.error('num_workers must be nonnegative and match_chunk positive.')
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.shots)) != len(args.shots):
        p.error('Datasets/shots cannot repeat.')
    if args.shots != sorted(args.shots):
        p.error('Shots must be in ascending order (1, 2, 4).')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        p.error('history_session must be one directory name.')
    worker = [args._session is not None, args._dataset is not None, args._shot is not None]
    if any(worker) and not all(worker):
        p.error('All internal worker fields are required together.')
    for name in ('visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def options(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')}


def prepare(args, dataset, shot):
    if dataset == 'Visa':
        source_dir = args.visa_source_root / f'visa_{shot}shot_seed10'
        control_dir = args.visa_control_root / f'visa_{shot}shot_seed10'
    else:
        source_dir = ROOT/'results/branch_diagnostics'/args.history_session/dataset/f'{shot}shot_seed10'
        control_dir = ROOT/'results/dual_output'/args.history_session/dataset/f'{shot}shot_seed10'
    source = audit_source(source_dir, shot, ROOT, expected_dataset=IDS[dataset])
    if source['config']['image_size'] != 518:
        raise ValueError('This predefined four-condition experiment requires image_size=518.')
    check_reference_separation(source)
    selection = select_samples([source], 1)
    control = audit_controls(source, control_dir)
    meta = Path(source['config']['test_data_path'])/'meta.json'
    audit = {'source': source['source'], 'control': control['directory'], 'source_hashes': source['hashes'],
             'control_hashes': control['hashes'], 'metadata_sha256': hashlib.sha256(meta.read_bytes()).hexdigest(),
             'selection': selection}
    return source, control, audit


def save_sample(args, source, output, sample, image, mask, raw, maps, align, index):
    """Selected final/ALIGN maps, no rescaling; compare A with original FINAL."""
    import numpy as np
    raw_arrays = {name: value[index].detach().cpu().numpy() for name, value in raw.items()}
    arrays = {name: value[index].cpu().numpy() for name, value in maps.items()}
    relative = Path(sample['Category'])/sample['Sample Type']/sample['Sample Key']
    if args.save_visualizations:
        save_panels(output/'visualizations'/relative.with_suffix('.png'), image, mask.cpu().numpy(),
                    raw_arrays, arrays, list(CONDITIONS), 3,
                    f"{source['config']['dataset']} / {source['config']['k_shots']}-shot / A B C D final maps")
    checks = []
    for stage, values, field in [('raw', raw_arrays, 'Raw Maps'), ('evaluated', arrays, 'Evaluated Maps')]:
        if args.save_maps:
            folder = output/'raw_maps'/relative/stage
            folder.mkdir(parents=True)
            for name, value in values.items():
                np.save(folder/f'{name}.npy', value, allow_pickle=False)
        old = Path(source['source'])/sample[field]/'FINAL.npy'
        if not old.is_file():
            raise FileNotFoundError(f'Required selected historical FINAL matrix missing: {old}')
        previous = np.load(old, allow_pickle=False)
        value = values['baseline']
        passed = previous.shape == value.shape and np.isfinite(previous).all() and np.allclose(
            previous, value, atol=1e-6, rtol=1e-5)
        checks.append({'Image Path': sample['Image Path'], 'Stage': stage, 'Passed': bool(passed),
                       'Max Abs Delta': float(np.max(np.abs(previous-value))) if previous.shape == value.shape else None})
    if args.save_maps:
        folder = output/'raw_maps'/relative/'align'
        folder.mkdir()
        for name, value in align.items():
            np.save(folder/f'{name}.npy', value[index].detach().cpu().numpy(), allow_pickle=False)
    return checks


def run_inference(args, source, control, output, logger):
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import smooth_maps
    from test import prompt_association
    from tools import setup_seed
    from tqdm import tqdm
    setup_seed(10)
    initialized, raw_memory, support_calls = initialize_both(source, output, args.num_workers, logger)
    config, model, textual, visual, pq, learned_text, memory, loader, dpam, device = initialized
    matcher = ContextMatcher(memory[1], raw_memory, args.match_chunk)
    tap = RawTap(model.visual.transformer)
    original_align = {}

    def observe_pqa(module, inputs, result):
        original_align['layers'] = [v.detach() for v in result[2]]

    handle = pq.register_forward_hook(observe_pqa)
    selected = {path_key(r['Image Path']): r for r in source['diagnostics']}
    pixels, images, score_rows, parity, matrix_checks, routing = [], [], [], [], [], []
    processed, visited, closed, buffer = 0, set(), set(), None

    def checkpoint():
        for filename, rows in [('image_scores.csv', score_rows), ('baseline_score_checks.csv', parity),
                               ('baseline_map_checks.csv', matrix_checks), ('routing_checks.csv', routing)]:
            if rows:
                write_csv(output/filename, list(rows[0]), rows)

    def verify(combined):
        checks = compare_baseline(source, combined, .001)
        write_csv(output/'baseline_reference_comparison.csv', list(checks[0]), checks)
        if not all(r['Passed'] for r in checks):
            raise RuntimeError('A differs from original full-precision Baseline metrics.')

    def flush():
        p_rows, i_rows = buffer.finish(device, logger)
        pixels.extend(p_rows)
        images.extend(i_rows)
        verify([r for r in metric_tables(output, pixels, images) if r['Category'] != 'Mean'])
        checkpoint()
        gc.collect()
        torch.cuda.empty_cache()

    try:
        for items in tqdm(loader, desc=f'{config.dataset} / {config.k_shots}-shot / A B C D'):
            paths, categories = items['img_path'], items['cls_name']
            with torch.no_grad():
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                raw_patches = tap.take()
                prompt_features, prompt_patches = prompt_association(*memory, categories)
                old_raw, old_maps, old_scores, maxima, globals_, fixed, extras = infer_batch(
                    config, features, patches, prompt_features, prompt_patches,
                    textual, visual, pq, learned_text, paths)
                versions = [v._version for v in [*patches, *raw_patches]]
                if args.disable_context_alignment:
                    align = {name: extras['ALIGN'] for name in CONDITIONS}
                else:
                    align = {'baseline': extras['ALIGN'], **matcher.alignments(
                        patches, raw_patches, categories, original_align['layers'], config.image_size)}
                raw, maps, scores = ({'baseline': v} for v in (old_raw['baseline'], old_maps['baseline'], old_scores['baseline']))
                for name in CONDITIONS[1:]:
                    if args.disable_context_alignment:
                        raw[name], maps[name], scores[name] = raw['baseline'], maps['baseline'], scores['baseline']
                    else:
                        raw[name] = fusion_fun([old_raw['no_align'], align[name]], fusion_type='harmonic_mean')
                        maps[name] = smooth_maps(raw[name], config.sigma).cpu()
                        maximum = maps[name].flatten(1).max(dim=1).values.to(device)
                        scores[name] = fusion_fun([fixed, maximum], fusion_type='harmonic_mean').cpu()
                untouched = versions == [v._version for v in [*patches, *raw_patches]]
                recovered = all(torch.equal(maps[n], maps['baseline']) and torch.equal(scores[n], scores['baseline'])
                                for n in CONDITIONS[1:]) if args.disable_context_alignment else None
                route = {'Batch': len(routing), 'Query Encoder Calls': 1, 'Feature Inputs Unmodified': untouched,
                         'Disable Requested': args.disable_context_alignment, 'Disabled Recovery': recovered,
                         'Raw Width': raw_patches[0].shape[-1], 'Projected Width': patches[0].shape[-1]}
                routing.append(route)
                if not untouched or recovered is False:
                    raise RuntimeError('Feature isolation/disabled recovery failed.')
                masks = (items['img_mask'][:, 0] > .5).to(torch.uint8)
                for name in CONDITIONS:
                    if maps[name].shape != masks.shape or scores[name].shape != (len(paths),):
                        raise ValueError('Unexpected candidate dimensions.')
                    if not torch.isfinite(raw[name]).all() or not torch.isfinite(maps[name]).all() or not torch.isfinite(scores[name]).all():
                        raise ValueError('Nonfinite candidate output.')
            for i, category in enumerate(categories):
                if processed >= len(source['manifest']):
                    raise ValueError('Extra query.')
                expected = source['manifest'][processed]
                label = int(items['anomaly'][i])
                if path_key(paths[i]) != path_key(expected['Image Path']) or category != expected['Category'] or label != int(expected['Sample Type']=='Anomaly'):
                    raise ValueError('Query identity/order/label mismatch.')
                if buffer is None or category != buffer.category:
                    if buffer is not None:
                        flush()
                        closed.add(buffer.category)
                    if category in closed:
                        raise ValueError('Non-contiguous category order.')
                    buffer = CategoryBuffer(category)
                buffer.append(masks[i], label, maps, scores, i)
                row = {'Category': category, 'Image Path': paths[i], 'Label': label,
                       **{f'{n} Global': float(v[i]) for n,v in globals_.items()},
                       'Fixed Global Mean': float(fixed[i]), 'baseline Map Max': float(maxima['baseline'][i]),
                       **{f'{n} Image Score': float(scores[n][i]) for n in CONDITIONS}}
                previous = control['scores'][processed]
                names = ['VA Global','TA Global','PQA Global','Fixed Global Mean','baseline Map Max','baseline Image Score']
                differences = [abs(row[n]-float(previous[n])) for n in names]
                passed = all(math.isclose(row[n], float(previous[n]), abs_tol=1e-6, rel_tol=1e-5) for n in names)
                parity.append({'Image Path': paths[i], 'Passed': passed, 'Max Abs Delta': max(differences)})
                score_rows.append(row)
                if not passed:
                    raise RuntimeError('A historical per-image score mismatch.')
                key = path_key(paths[i])
                if key in selected:
                    checks = save_sample(args, source, output, selected[key], items['img'][i], masks[i], raw, maps, align, i)
                    matrix_checks.extend(checks)
                    visited.add(key)
                    if not all(r['Passed'] for r in checks):
                        raise RuntimeError('A historical FINAL matrix mismatch.')
                processed += 1
            del features, patches, raw_patches, prompt_features, prompt_patches, old_raw, old_maps, old_scores, raw, maps, scores, align, extras
            original_align.clear()
        if buffer is None or processed != len(source['manifest']) or visited != set(selected):
            raise ValueError('Incomplete query/example coverage.')
        flush()
        combined = metric_tables(output, pixels, images)
        verify(combined)
        write_csv(output/'metrics_combined.csv', ['Category','Condition',*FIELDS], combined)
        return {'status':'completed','complete':True,'sample_count':processed,'category_count':len(closed)+1,
                'conditions':list(CONDITIONS),'query_encoder_calls':len(routing),'support_encoder_calls':support_calls,
                'selected_samples':len(visited),'disable_context_alignment':args.disable_context_alignment}
    finally:
        try:
            checkpoint()
        finally:
            tap.close()
            handle.remove()


def summarize(session, completed):
    rows, effects = [], []
    for dataset, shot in completed:
        combined = read_csv(session/dataset/f'{shot}shot_seed10'/'metrics_combined.csv')
        index = {(r['Category'],r['Condition']):r for r in combined}
        rows.extend({'Dataset':dataset,'Shot':shot,**r} for r in combined)
        for category in {r['Category'] for r in combined}:
            for candidate, reference in PAIRS:
                effects.append({'Dataset':dataset,'Shot':shot,'Category':category,'Condition':candidate,
                    'Comparator':reference,**{f:float(index[(category,candidate)][f])-float(index[(category,reference)][f]) for f in FIELDS}})
    columns=['Dataset','Shot','Category','Condition',*FIELDS]
    write_csv(session/'metrics_summary.csv',columns,[r for r in rows if r['Category']=='Mean'])
    write_csv(session/'metrics_by_category.csv',columns,[r for r in rows if r['Category']!='Mean'])
    write_csv(session/'factorial_effects.csv',['Dataset','Shot','Category','Condition','Comparator',*FIELDS],effects)
    # Interaction asks whether the neighborhood effect changes with interface.
    interactions = []
    for dataset,shot in completed:
        table = {(r['Category'],r['Condition']):r for r in rows if r['Dataset']==dataset and r['Shot']==shot}
        for category in {key[0] for key in table}:
            a,b,c,d = [table[(category,name)] for name in CONDITIONS]
            interactions.append({'Dataset':dataset,'Shot':shot,'Category':category,
                **{f:float(d[f])-float(c[f])-float(b[f])+float(a[f]) for f in FIELDS}})
    write_csv(session/'interaction_effects.csv',['Dataset','Shot','Category',*FIELDS],interactions)
    retention=[]
    for shot in sorted({s for _,s in completed}):
        for condition in CONDITIONS[1:]:
            group=[r for r in effects if r['Shot']==shot and r['Category']=='Mean'
                   and r['Condition']==condition and r['Comparator']=='baseline']
            nets=[sum(r[f] for f in FIELDS) for r in group]
            complete={r['Dataset'] for r in group}==set(DATASETS)
            retention.append({'Shot':shot,'Condition':condition,'Datasets':len(group),'Complete Six':complete,
                'Improved Datasets':sum(v>0 for v in nets),'Net Total':sum(nets),
                'Pass':(sum(v>0 for v in nets)>=3 and sum(nets)>0) if complete else None})
    write_csv(session/'retention_summary.csv',list(retention[0]),retention)


def run_worker(args):
    session = Path(args._session).resolve()
    session.relative_to(ROOT/'results'/Path(__file__).stem)
    saved = read_json(session/'campaign_config.json')
    if saved['options'] != options(args) or args._dataset not in args.datasets or args._shot not in args.shots:
        raise ValueError('Worker plan/options changed.')
    output = session/args._dataset/f'{args._shot}shot_seed10'
    output.mkdir(parents=True)
    logger = logging.getLogger('context_alignment')
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output/'run.log',encoding='utf-8'),logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    try:
        write_json(output/'run_status.json',{'status':'running','complete':False})
        source, control, audit = prepare(args,args._dataset,args._shot)
        if audit != read_json(session/'input_audit.json')[args._dataset][str(args._shot)]:
            raise ValueError('Inputs changed since parent audit.')
        import torch
        import numpy as np
        import scipy
        write_json(output/'experiment_config.json',{**source['config'],**options(args),
            'result_dir':str(output),'source_result_dir':source['source'],'control_result_dir':control['directory'],
            'conditions':list(CONDITIONS),'kernels':[1,3],'scale_weight':.5,'pool_padding':'constant zero; full denominator',
            'raw_policy':'pre-ln_post/proj 1024; same AdaptCLIP DPAM forward; not original DictAS backbone',
            'projected_policy':'unchanged ln_post/proj 768; adapters always use this',
            'candidate_policy':'replace final ALIGN only; PQA internal matching unchanged',
            'image_policy':'original harmonic(fixed native global mean, max(smoothed final map))',
            'versions':{'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__}})
        write_json(output/'input_audit.json',audit)
        (output/'reference_samples.txt').write_text(''.join(f'{c}\t{p}\n' for c,p in source['reference_paths']),encoding='utf-8')
        write_csv(output/'source_sample_manifest.csv',list(source['manifest'][0]),source['manifest'])
        write_csv(output/'visualization_samples.csv',list(audit['selection']['samples'][0]),audit['selection']['samples'])
        status = run_inference(args,source,control,output,logger)
        status.update(dataset=IDS[args._dataset],k_shots=args._shot,seed=10)
        write_json(output/'run_status.json',status)
    except BaseException as error:
        write_json(output/'run_status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                                            'complete':False,'error':str(error)})
        logger.exception('Round stopped.')
        raise
    finally:
        gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            sys.modules['torch'].cuda.empty_cache()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main(argv=None):
    args = parse_args(argv)
    if args._session:
        run_worker(args)
        return
    inputs = {}
    jobs = [(d,s) for d in args.datasets for s in args.shots]
    for dataset,shot in jobs:
        _,_,audit = prepare(args,dataset,shot)
        inputs.setdefault(dataset,{})[str(shot)] = audit
    if args.dry_run:
        print(f'No inference/output. Plan: {jobs}; four conditions per round.')
        return
    session = new_directory(ROOT/'results'/Path(__file__).stem,'six_datasets_seed10')
    write_json(session/'campaign_config.json',{'options':options(args),'conditions':list(CONDITIONS),'seed':10})
    write_json(session/'input_audit.json',inputs)
    write_json(session/'execution_plan.json',[{'Dataset':d,'Shot':s} for d,s in jobs])
    completed,current = [],None
    try:
        for i,(dataset,shot) in enumerate(jobs,1):
            current=[dataset,shot]
            write_json(session/'run_status.json',{'status':'running','complete':False,'completed_rounds':len(completed),
                                                 'total_rounds':len(jobs),'current_round':current})
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--_session',str(session),
                '--_dataset',dataset,'--_shot',str(shot),'--datasets',*args.datasets,'--shots',*[str(s) for s in args.shots]]
            for name in ('history_session','visa_source_root','visa_control_root','num_workers','match_chunk'):
                command.extend([f'--{name}',str(getattr(args,name))])
            for name in ('disable_context_alignment','save_maps'):
                if getattr(args,name):
                    command.append(f'--{name}')
            if not args.save_visualizations:
                command.append('--no-save_visualizations')
            print(f'[{i}/{len(jobs)}] {dataset} {shot}-shot; output={session}',flush=True)
            with subprocess.Popen(command,cwd=str(ROOT)) as child:
                try:
                    code=child.wait()
                except KeyboardInterrupt:
                    try:
                        child.wait(timeout=15)
                    except (subprocess.TimeoutExpired,KeyboardInterrupt):
                        if child.poll() is None:
                            child.terminate()
                        child.wait()
                    raise
            if code or read_json(session/dataset/f'{shot}shot_seed10'/'run_status.json').get('complete') is not True:
                raise RuntimeError(f'Incomplete/failed round: {current}; remaining rounds stopped.')
            completed.append((dataset,shot))
            summarize(session,completed)
        write_json(session/'run_status.json',{'status':'completed','complete':True,'completed_rounds':len(completed),
                                             'total_rounds':len(jobs),'completed_jobs':completed})
    except BaseException as error:
        write_json(session/'run_status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            'complete':False,'completed_rounds':len(completed),'total_rounds':len(jobs),'failed_round':current,'error':str(error)})
        raise
    print(f'Completed. {session / "metrics_summary.csv"}',flush=True)


if __name__ == '__main__':
    main()
