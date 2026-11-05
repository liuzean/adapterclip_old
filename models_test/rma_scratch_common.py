"""Shared FROM-SCRATCH adapter training; no old adapter states are loaded.

original: official data/sampling/Adam/loss/epoch protocol plus RMA supervision.
new: held-out source validation, mixed references, AdamW/accumulation/fusion loss.
The separate train_new_d_three entry selects the matched three-adapter control.
"""
import argparse
import logging
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json, write_csv, write_json
from rma_joint_utils import (FullBatchSampler, adapter_states, encode_learned_text,
                             evaluate_candidate, joint_forward, joint_losses, require_joint_gradients)
from rma_utils import (FIELDS, MetricsBuffer, atomic_checkpoint, check_frozen, cleanup,
                       curves, digest, failure_status, freeze_base, save_tables, source_split,
                       validate_checkpoint_payload, record_pipeline_progress)

FORMAT = 'adaptclip-rma-scratch-v1'
THREE_FORMAT = 'adaptclip-three-scratch-v1'
THREE_INITIALIZATION = 'CLIP pretrained; three adapters fresh; no RMA'
PROFILES = {
    'original': dict(batch_size=8, accumulation=1, learning_rate=.001, weight_decay=0.,
                     betas=[.5, .999], optimizer='adam', train_reference_shots=[1],
                     fusion_weight=0., gradient_clip=0., num_workers=4),
    'new': dict(batch_size=2, accumulation=4, learning_rate=.0001, weight_decay=.0001,
                betas=[.9, .999], optimizer='adamw', train_reference_shots=[1, 2, 4],
                fusion_weight=1., gradient_clip=1., num_workers=0),
}


def parse_training(flow, argv=None, *, three_adapter_control=False):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--train_data_path', default='E:/projects/datasets/mvtec')
    p.add_argument('--pretrained_model', default='E:/projects/FSAD/AdaptCLIP/checkpoint/train_base/ViT-L-14-336px.pt')
    p.add_argument('--baseline_checkpoint', default='checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth',
                   help='Comparator identity ONLY, NEVER loaded for scratch training.')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--accumulation', type=int)
    p.add_argument('--learning_rate', type=float)
    p.add_argument('--weight_decay', type=float)
    p.add_argument('--num_workers', type=int)
    p.add_argument('--validation_fraction', type=float, default=.2)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--match_chunk', type=int, default=128)
    p.add_argument('--matching', choices=('learned', 'uniform', 'top1'), default='learned')
    if three_adapter_control:
        p.add_argument('--matched_new_d_run', default='results/train_rma_scratch_new/mvtec_seed10',
                       help='Completed new-D training whose settings, split and episodes must match.')
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.flow, args.seed = flow, 10
    args.three_adapter_control = three_adapter_control
    if three_adapter_control and flow != 'new':
        p.error('The matched three-adapter control uses the new-D flow only.')
    args.overrides = []
    for key, value in PROFILES[flow].items():
        if hasattr(args, key) and getattr(args, key) is not None:
            if getattr(args, key) != value:
                args.overrides.append(key)
        else:
            setattr(args, key, value)
    if min(args.epochs, args.accumulation, args.top_k, args.match_chunk) < 1 or args.batch_size < 2 or args.num_workers < 0:
        p.error('Positive sizes/epochs required; batch >=2 and workers >=0.')
    if args.hidden_dim < 8 or args.hidden_dim % 8:
        p.error('hidden_dim must be divisible by 8.')
    if not all(math.isfinite(x) for x in (args.learning_rate, args.weight_decay, args.validation_fraction)):
        p.error('Settings must be finite.')
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 < args.validation_fraction < 1:
        p.error('Invalid optimizer or validation settings.')
    for name in ('train_data_path', 'pretrained_model', 'baseline_checkpoint'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    if three_adapter_control:
        args.matched_new_d_run = project_path(args.matched_new_d_run, ROOT)
    return args


def protocol(args):
    config = read_json(ROOT/'results/branch_diagnostics/visa_1shot_seed10/experiment_config.json')
    config.update(dataset='mvtec', test_data_path=str(args.train_data_path), pretrained_model=str(args.pretrained_model),
                  checkpoint_path=str(args.baseline_checkpoint), seed=10, batch_size=args.batch_size, k_shots=1)
    if config['features_list'] != [6, 12, 18, 24] or config['image_size'] != 518:
        raise ValueError('Scratch protocol requires the agreed four layers and image_size518.')
    if not all(config[k] for k in ('visual_learner','textual_learner','pq_learner','pq_context')):
        raise ValueError('Scratch comparison requires all three original adapters and pq_context enabled.')
    return SimpleNamespace(**config)


def initialize_fresh(args, config):
    """Construct fresh TA/VA/PQA and optionally RMA in the original order.

    CLIP stays frozen while its text path differentiates with respect to prompts.
    RMA initialization RNG is restored so it does not shift original sampling RNG.
    """
    import torch
    import adaptcliplib
    from adaptcliplib import TextualAdapter, VisualAdapter, PQAdapter
    from rma_model import ReferenceMatchingAdapter
    from train import resolve_pretrained_model_type
    if not torch.cuda.is_available():
        raise RuntimeError('Original static text construction requires CUDA.')
    # The original helper calls str.replace; pathlib.Path.replace means rename.
    if resolve_pretrained_model_type(str(args.pretrained_model)) != 'ViT-L/14@336px':
        raise ValueError('This scratch experiment currently supports ViT-L/14@336px only.')
    model, _ = adaptcliplib.load(str(args.pretrained_model), device='cuda')
    model.visual.DAPM_replace(DPAM_layer=20)
    adapters = (TextualAdapter(model.to('cpu'), config.image_size, config.n_ctx),
                VisualAdapter(config.image_size, 14, input_dim=768, reduction=config.vl_reduction),
                PQAdapter(config.image_size, 14, context=config.pq_context, input_dim=768,
                          mid_dim=config.pq_mid_dim, layers_num=4))
    model.to('cuda').eval().requires_grad_(False)
    for module in adapters:
        module.to('cuda').train().requires_grad_(True)
    rma = None
    if not getattr(args, 'three_adapter_control', False):
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        rma = ReferenceMatchingAdapter(hidden_dim=args.hidden_dim, top_k=args.top_k,
            image_size=config.image_size, chunk=args.match_chunk, matching=args.matching).to('cuda')
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    adapters[0].prepare_static_text_feature(model)
    return model, adapters, rma


def source_validation(config, model, adapters, rma, loader, memory, output, epoch):
    """Source-held-out full-model score; the three-control never calls RMA."""
    import torch
    from test import prompt_association
    for module in (*adapters, rma) if rma is not None else adapters:
        module.eval()
    rows = []
    with torch.no_grad():
        text = encode_learned_text(model, adapters[0])
        for shot in (1, 2, 4):
            pixels, images, buffer = [], [], None
            directory = output/'validation'/f'epoch_{epoch:03d}'/f'{shot}shot'
            directory.mkdir(parents=True)
            for items in loader:
                features, patches = model.encode_image(items['img'].cuda(), config.features_list, DPAM_layer=20)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf, pp = pf[:, :shot].cuda(), [p[:, :shot].cuda() for p in pp]
                _, maps, scores = evaluate_candidate(config, adapters, text, rma, features, patches, pf, pp,
                                                     items['img_path'], without_rma=rma is None)
                for i, category in enumerate(items['cls_name']):
                    if buffer is None or buffer.category != category:
                        if buffer is not None:
                            p, q = buffer.finish('cuda')
                            pixels.extend(p)
                            images.extend(q)
                        buffer = MetricsBuffer(category, conditions=('scratch',))
                    buffer.append(items['img_mask'][i, 0] > .5, items['anomaly'][i], {'scratch': maps}, {'scratch': scores}, i)
            if buffer is None:
                raise ValueError('Empty source validation.')
            p, q = buffer.finish('cuda')
            pixels.extend(p)
            images.extend(q)
            rows.extend(dict(epoch=epoch, shot=shot, **r) for r in save_tables(directory, pixels, images))
            cleanup()
    means = [r for r in rows if r['Category'] == 'Mean']
    if len(means) != 3 or any(not math.isfinite(float(r[f])) for r in means for f in FIELDS):
        raise ValueError('Invalid source validation score.')
    return sum(float(r[f]) for r in means for f in FIELDS)/18, rows


def validate_scratch(payload, base_hash, clip_hash, config, flow):
    if payload.get('format') != FORMAT or payload.get('flow') != flow or payload.get('initialization') != 'CLIP pretrained; all four adapters fresh':
        raise ValueError('Use the matching FROM-SCRATCH flow checkpoint, not B or warm-start C.')
    if any(k not in payload for k in ('textual_learner', 'visual_learner', 'pq_learner', 'rma')):
        raise ValueError('All four adapter states required.')
    validate_checkpoint_payload(dict(payload, format='adaptclip-rma-v1'), base_hash, clip_hash, config)


def validate_three_scratch(payload, base_hash, clip_hash, config, flow):
    """Reject four-adapter/old-B checkpoints for the independently trained control."""
    if (flow != 'new' or payload.get('format') != THREE_FORMAT or payload.get('flow') != flow
            or payload.get('initialization') != THREE_INITIALIZATION or 'rma' in payload
            or any(k not in payload for k in ('textual_learner', 'visual_learner', 'pq_learner'))):
        raise ValueError('Use the new-flow three-adapter control checkpoint, not new D with RMA removed.')
    validate_checkpoint_payload(dict(payload, format='adaptclip-rma-v1'), base_hash, clip_hash, config)


def training_losses(flow, globals_, pixels, raw, labels, masks, focal, dice, fusion_weight):
    """Original flow keeps branch-only supervision; new flow adds fusion loss."""
    if flow == 'new':
        return joint_losses(globals_, pixels, raw, labels, masks, focal, dice, fusion_weight)
    from torch.nn import functional as F
    global_sum, local_sum, parts = 0, 0, {}
    for name in ('VA', 'TA', 'PQA', 'RMA'):
        g = sum(F.cross_entropy(v, labels.long()) for v in globals_[name])
        local = 0
        for v in pixels[name]:
            local = local + focal(v, masks)
            local = local + dice(v[:, 1], masks)
            local = local + dice(v[:, 0], 1-masks)
        parts.update({f'{name}_global_loss':g, f'{name}_local_loss':local})
        global_sum, local_sum = global_sum+g, local_sum+local
    total = global_sum+local_sum
    return total, dict(global_loss=global_sum, local_loss=local_sum, fusion_loss=total.new_zeros(()), total_loss=total, **parts)


def checkpoint_selection(flow, summary, previous):
    """Original minimizes epoch mean global+local; new maximizes source validation.

    Strict comparisons retain the earlier epoch on an exact tie. Training loss
    covers all four adapters and is never interpreted as a held-out score.
    """
    value = (float(summary['global_loss']) + float(summary['local_loss'])
             if flow == 'original' else float(summary['validation_score']))
    if not math.isfinite(value):
        raise ValueError('Checkpoint selection value must be finite.')
    return value, value < previous if flow == 'original' else value > previous


def train(args, output, logger):
    import torch
    from adaptcliplib import BinaryDiceLoss, FocalLoss
    from dataset import Dataset
    from tools import get_transform, setup_seed
    from fusion_ablation_utils import RecordedPromptDataset
    from test import build_prompt_memory
    from tqdm import tqdm
    setup_seed(10)
    config = protocol(args)
    model, adapters, rma = initialize_fresh(args, config)
    frozen = freeze_base([None, model])
    modules = (*adapters, rma) if rma is not None else adapters
    parameters = [p for m in modules for p in m.parameters()]
    optimizer_class = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.AdamW
    optimizer = optimizer_class(parameters, lr=args.learning_rate, betas=tuple(args.betas), weight_decay=args.weight_decay)
    check_frozen(frozen, optimizer)
    transform, target_transform = get_transform(image_size=config.image_size)
    data = Dataset(root=str(args.train_data_path), transform=transform, target_transform=target_transform,
                   dataset_name='mvtec', k_shots=0, save_dir=str(output), mode='train' if args.flow == 'original' else 'test', seed=10)
    meta = read_json(args.train_data_path/'meta.json')
    if any(int(r['anomaly']) != 0 for pool in meta['train'].values() for r in pool):
        raise ValueError('Source reference pools must contain only normal samples.')
    write_json(output/'training_reference_pool.json', meta['train'])
    split, val_loader, memory = None, None, None
    if args.flow == 'new':
        split = source_split(args.train_data_path, 10, args.validation_fraction)
        write_json(output/'source_split.json', split)
        import copy
        val_data = copy.copy(data)
        val_data.data_all, val_data.length = split['validation_queries'], len(split['validation_queries'])
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        data.data_all, data.length = split['train_queries'], len(split['train_queries'])
        supports = RecordedPromptDataset(str(args.train_data_path), split['validation_supports'], split['classes'], transform, target_transform)
        support_loader = torch.utils.data.DataLoader(supports, batch_size=args.batch_size, shuffle=False)
        memory = build_prompt_memory(model, support_loader, 'cuda', data.obj_list, data.view_list, config.features_list, 20)
        memory = ({k:v.cpu() for k,v in memory[0].items()}, {k:[v.cpu() for v in vs] for k,vs in memory[1].items()})
        sampler = FullBatchSampler(torch.utils.data.RandomSampler(data), args.batch_size)
        loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, num_workers=args.num_workers)
    else:
        if len(data) % args.batch_size == 1:
            raise ValueError('Original-flow trailing batch has one query; choose a documented batch setting for BN.')
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    write_json(output/'training_queries.json', data.data_all)
    expected_episodes = None
    if args.three_adapter_control:
        from new_d_control_utils import check_training_inputs
        expected_episodes = check_training_inputs(args, output)
    metadata = dict(format=FORMAT if rma is not None else THREE_FORMAT, flow=args.flow,
        initialization='CLIP pretrained; all four adapters fresh' if rma is not None else THREE_INITIALIZATION,
        training_scheme=f'scratch_{args.flow}' if rma is not None else 'new_flow_three_adapter_control',
        model_spec=rma.spec if rma is not None else dict(input_dim=768, image_size=config.image_size), base_protocol=vars(config),
        base_sha256=digest(args.baseline_checkpoint), pretrained_sha256=digest(args.pretrained_model),
        train_options={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_meta_sha256=digest(args.train_data_path/'meta.json'), seed=10,
        matched_new_flow_three_adapter_control=rma is None,
        source_validation_used=args.flow == 'new',
        validation_fraction_used=args.validation_fraction if args.flow == 'new' else None,
        comparator_checkpoint_role='original A identity only; NEVER loaded into training',
        selection_rule='minimum epoch mean global_loss + local_loss; earliest on tie; no validation selection' if args.flow == 'original' else 'best held-out source six-metric / 1,2,4-shot mean')
    write_json(output/'model_manifest.json', dict(**metadata, train_query_count=len(data),
        validation_query_count=0 if val_loader is None else len(val_loader.dataset), frozen_modules=['CLIP'],
        trainable_modules=['VA','TA','PQA','RMA'] if rma is not None else ['VA','TA','PQA'],
        trainable_parameters=sum(p.numel() for p in parameters)))
    focal, dice, rng = FocalLoss(), BinaryDiceLoss(), random.Random(10)
    history, steps, episodes, validation_rows = [], [], [], []
    best_value, best_epoch = (math.inf if args.flow == 'original' else -math.inf), None
    for epoch in range(1, args.epochs+1):
        for m in modules:
            m.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_rows = []
        for step, items in enumerate(tqdm(loader, desc=f'Scratch {args.flow} {epoch}/{args.epochs}')):
            batch = len(items['img_path'])
            if args.flow == 'original':
                prompt_images = items['prompt_img'].cuda()
                shot = prompt_images.shape[1]
                prompt_images = prompt_images.flatten(0, 1)
                if shot != 1:
                    raise ValueError('Original Dataset must return one normal reference per query.')
            else:
                shot = args.train_reference_shots[len(steps) % len(args.train_reference_shots)]
                refs = []
                for path, category in zip(items['img_path'], items['cls_name']):
                    chosen = rng.sample(split['training_support_pool'][category], shot)
                    refs.extend(chosen)
                    episodes.append(dict(epoch=epoch, step=step+1, query=path, category=category, shot=shot,
                                         references=' | '.join(r['img_path'] for r in chosen)))
                    if expected_episodes is not None:
                        if {k:str(v) for k,v in episodes[-1].items()} != expected_episodes[len(episodes)-1]:
                            raise ValueError('Training episode differs from matched new D; no matched-control claim is valid.')
                support_data = RecordedPromptDataset(str(args.train_data_path), refs, split['classes'], transform, target_transform)
                prompt_images = torch.stack([support_data[i]['img'] for i in range(len(support_data))]).cuda()
            with torch.no_grad():
                features, patches = model.encode_image(items['img'].cuda(), config.features_list, DPAM_layer=20)
                pf, pp = model.encode_image(prompt_images, config.features_list, DPAM_layer=20)
                pf, pp = pf.reshape(batch, shot, -1), [p.reshape(batch, shot, *p.shape[1:]) for p in pp]
            globals_, pixels, raw = joint_forward(model, adapters, rma, features, patches, pf, pp)
            masks = (items['img_mask'][:, 0] > .5).float().cuda()
            loss, parts = training_losses(args.flow, globals_, pixels, raw, items['anomaly'].cuda(), masks, focal, dice, args.fusion_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss.')
            size = min(args.accumulation, len(loader)-(step//args.accumulation)*args.accumulation)
            (loss/size).backward()
            if (step+1) % args.accumulation == 0 or step+1 == len(loader):
                require_joint_gradients(adapters, rma)
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            row = dict(epoch=epoch, step=step+1, shot=shot, batch_size=batch, learning_rate=args.learning_rate,
                       **{k:float(v.detach()) for k,v in parts.items()})
            steps.append(row)
            epoch_rows.append(row)
            del loss, parts, globals_, pixels, raw, prompt_images, features, patches, pf, pp
        check_frozen(frozen, optimizer)
        score = None
        if args.flow == 'new':
            score, val_rows = source_validation(config, model, adapters, rma, val_loader, memory, output, epoch)
            validation_rows.extend(val_rows)
            write_csv(output/'validation_metrics.csv', ['epoch','shot','Category','Condition',*FIELDS], validation_rows)
        summary = dict(epoch=epoch, **{k:sum(r[k] for r in epoch_rows)/len(epoch_rows)
                       for k in epoch_rows[0] if k not in ('epoch','step','shot','batch_size')}, validation_score=score)
        history.append(summary)
        selection_value, improved = checkpoint_selection(args.flow, summary, best_value)
        if improved:
            best_value, best_epoch = selection_value, epoch
        payload = dict(**metadata, **adapter_states(adapters), **({'rma':rma.state_dict()} if rma is not None else {}), epoch=epoch,
                       validation_score=score, best_epoch=best_epoch,
                       best_score=best_value if args.flow == 'new' else None,
                       best_training_loss=best_value if args.flow == 'original' else None)
        atomic_checkpoint(output/f'epoch_{epoch}.pth', payload)
        if improved:
            atomic_checkpoint(output/'best.pth', payload)
        atomic_checkpoint(output/'last.pth', dict(**payload, optimizer=optimizer.state_dict()))
        write_csv(output/'training_history.csv', list(history[0]), history)
        write_csv(output/'training_steps.csv', list(steps[0]), steps)
        if episodes:
            write_csv(output/'training_episodes.csv', list(episodes[0]), episodes)
        # Original-flow has no validation score; plot its losses without inventing one.
        if args.flow == 'new':
            curves(output, history)
        logger.info('epoch=%s global=%.6f local=%.6f total=%.6f validation=%s best_epoch=%s selection_value=%.6f', epoch,
                    summary['global_loss'], summary['local_loss'], summary['total_loss'], score, best_epoch, best_value)
        write_json(output/'run_status.json', dict(status='running', complete=False, epoch=epoch, flow=args.flow,
            best_epoch=best_epoch, selected_checkpoint=str(output/'best.pth'),
            best_training_loss=best_value if args.flow == 'original' else None,
            best_score=best_value if args.flow == 'new' else None))
        cleanup()
    selected = output/'best.pth'
    if expected_episodes is not None and len(episodes) != len(expected_episodes):
        raise ValueError('Incomplete matched new-D episode coverage.')
    return dict(status='completed', complete=True, epochs=args.epochs, flow=args.flow,
                selected_checkpoint=str(selected), best_epoch=best_epoch,
                best_training_loss=best_value if args.flow == 'original' else None,
                best_score=best_value if args.flow == 'new' else None,
                frozen_integrity_passed=True,
                **({'four_adapter_gradient_checks_passed':True} if rma is not None else {'three_adapter_gradient_checks_passed':True}))


def training_main(flow, entry_stem, argv=None, *, three_adapter_control=False):
    args = parse_training(flow, argv, three_adapter_control=three_adapter_control)
    for path in (args.train_data_path/'meta.json', args.pretrained_model, args.baseline_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if three_adapter_control:
        from new_d_control_utils import check_training_settings
        check_training_settings(args)
    output = new_directory(ROOT/'results'/entry_stem, 'mvtec_seed10')
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    record_pipeline_progress(receipt, 'training', session=str(output), flow=flow, epochs=args.epochs)
    logger = logging.getLogger(entry_stem)
    logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(output/'train.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        logger.addHandler(handler)
    write_json(output/'experiment_config.json', {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    write_json(output/'run_status.json', dict(status='initializing', complete=False, flow=flow))
    try:
        state = train(args, output, logger)
        write_json(output/'run_status.json', state)
        record_pipeline_progress(receipt, 'completed', session=str(output), flow=flow,
                                 selected_checkpoint=state['selected_checkpoint'], epochs=args.epochs)
        print(f'Training completed. Test checkpoint: {state["selected_checkpoint"]}', flush=True)
    except BaseException as error:
        failure_status(output, error)
        logger.exception('Scratch training stopped.')
        raise
    finally:
        cleanup()
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
