"""C: jointly train VA/TA/PQA/RMA from epoch_15, keeping CLIP frozen."""
import argparse
import copy
import logging
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, write_csv, write_json
from rma_joint_utils import (
    JOINT, JointBuffer, FullBatchSampler, adapter_states, clone_adapters, encode_learned_text,
    evaluate_candidate, joint_forward, joint_losses, joint_validation_score,
    require_joint_gradients, verify_joint_optimizer,
)
from rma_utils import (
    FIELDS, atomic_checkpoint, check_frozen, cleanup, curves, digest, failure_status,
    freeze_base, save_tables, source_spec, source_split,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--train_data_path', default='E:/projects/datasets/mvtec')
    p.add_argument('--base_checkpoint', default='checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth')
    p.add_argument('--pretrained_model', default='E:/projects/FSAD/AdaptCLIP/checkpoint/train_base/ViT-L-14-336px.pt')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=1e-4, help='New RMA learning rate.')
    p.add_argument('--adapter_learning_rate', type=float, default=1e-5, help='VA/TA/PQA fine-tuning learning rate.')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--validation_fraction', type=float, default=.2)
    p.add_argument('--fusion_weight', type=float, default=1.)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--match_chunk', type=int, default=128)
    p.add_argument('--matching', choices=('learned', 'uniform', 'top1'), default='learned')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--save_every_epoch', action='store_true')
    p.add_argument('--_output', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if min(args.epochs, args.accumulation, args.top_k, args.match_chunk) < 1 or args.batch_size < 2 or args.num_workers < 0:
        p.error('Joint batch_size >= 2; epochs/sizes/accumulation positive; workers nonnegative.')
    if args.hidden_dim < 8 or args.hidden_dim % 8:
        p.error('hidden_dim must be divisible by eight.')
    if any(not math.isfinite(v) for v in (args.learning_rate, args.adapter_learning_rate,
            args.weight_decay, args.validation_fraction, args.fusion_weight)):
        p.error('Optimizer/loss settings must be finite.')
    if min(args.learning_rate, args.adapter_learning_rate) <= 0 or min(args.weight_decay, args.fusion_weight) < 0:
        p.error('Learning rates must be positive; weight decay/fusion weight nonnegative.')
    if not 0 < args.validation_fraction < 1:
        p.error('validation_fraction must be between zero and one.')
    for name in ('train_data_path', 'base_checkpoint', 'pretrained_model'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def validation(initialized, adapters, rma, output, epoch):
    """Select C with fixed source queries/supports and its current FOUR states."""
    import torch
    from fusion_ablation import infer_batch
    from test import prompt_association
    config, model, textual, visual, pq, original_text, memory, loader, dpam, device = initialized
    for module in (*adapters, rma):
        module.eval()
    tables = {}
    with torch.no_grad():
        candidate_text = encode_learned_text(model, adapters[0])
        for shot in (1, 2, 4):
            directory = output/'validation'/f'epoch_{epoch:03d}'/f'{shot}shot'
            directory.mkdir(parents=True)
            pixels, images, buffer = [], [], None
            for items in loader:
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf, pp = pf[:, :shot].to(device), [p[:, :shot].to(device) for p in pp]
                _, old_maps, old_scores, _, _, _, _ = infer_batch(
                    config, features, patches, pf, pp, textual, visual, pq, original_text, items['img_path'])
                _, candidate_map, candidate_score = evaluate_candidate(
                    config, adapters, candidate_text, rma, features, patches, pf, pp, items['img_path'])
                maps = {'baseline': old_maps['baseline'], JOINT: candidate_map}
                scores = {'baseline': old_scores['baseline'], JOINT: candidate_score}
                for i, category in enumerate(items['cls_name']):
                    if buffer is None or buffer.category != category:
                        if buffer is not None:
                            p, q = buffer.finish(device)
                            pixels.extend(p)
                            images.extend(q)
                        buffer = JointBuffer(category)
                    buffer.append(items['img_mask'][i, 0] > .5, items['anomaly'][i], maps, scores, i)
            if buffer is None:
                raise ValueError('Empty source validation loader.')
            p, q = buffer.finish(device)
            pixels.extend(p)
            images.extend(q)
            tables[shot] = save_tables(directory, pixels, images)
            cleanup()
    return joint_validation_score(tables), tables


def train(args, output, logger):
    import torch
    from adaptcliplib import BinaryDiceLoss, FocalLoss
    from fusion_ablation import initialize
    from fusion_ablation_utils import RecordedPromptDataset
    from rma_model import ReferenceMatchingAdapter
    from tools import setup_seed
    from tqdm import tqdm
    setup_seed(10)
    split = source_split(args.train_data_path, 10, args.validation_fraction)
    write_json(output/'source_split.json', split)
    spec = source_spec(args, split)
    write_json(output/'base_protocol.json', spec['config'])
    initialized = list(initialize(spec, output, args.num_workers, logger))
    config, model, _, _, _, _, memory, val_loader, dpam, device = initialized
    initialized[6] = ({k: v.cpu() for k, v in memory[0].items()},
                      {k: [v.cpu() for v in values] for k, values in memory[1].items()})
    del memory
    validation_dataset = val_loader.dataset
    validation_dataset.data_all = split['validation_queries']
    validation_dataset.length = len(validation_dataset.data_all)
    training_dataset = copy.copy(validation_dataset)
    training_dataset.data_all = split['train_queries']
    training_dataset.length = len(training_dataset.data_all)
    sampler = FullBatchSampler(torch.utils.data.RandomSampler(training_dataset), args.batch_size)
    loader = torch.utils.data.DataLoader(training_dataset, batch_sampler=sampler, num_workers=args.num_workers)
    frozen = freeze_base(initialized)  # CLIP and original A copies remain read-only.
    adapters = clone_adapters(initialized, trainable=True)
    rma = ReferenceMatchingAdapter(hidden_dim=args.hidden_dim, top_k=args.top_k,
        image_size=config.image_size, chunk=args.match_chunk, matching=args.matching).to(device)
    old_parameters = [p for module in adapters for p in module.parameters()]
    optimizer = torch.optim.AdamW([
        dict(params=old_parameters, lr=args.adapter_learning_rate),
        dict(params=list(rma.parameters()), lr=args.learning_rate)], weight_decay=args.weight_decay)
    verify_joint_optimizer(frozen, adapters, rma, optimizer)
    focal, dice = FocalLoss(), BinaryDiceLoss()
    metadata = dict(format='adaptclip-rma-joint-v1', training_scheme=JOINT, model_spec=rma.spec,
        base_protocol=spec['config'], base_sha256=digest(args.base_checkpoint),
        pretrained_sha256=digest(args.pretrained_model), source_split_sha256=digest(output/'source_split.json'),
        train_options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')},
        seed=10, selection_rule='C full-model source validation; category/six metrics/1,2,4-shot equal mean')
    write_json(output/'model_manifest.json', dict(**metadata,
        trainable_modules=['VA', 'TA', 'PQA', 'RMA'], frozen_modules=['CLIP', 'original A adapters'],
        trainable_parameters=sum(p.numel() for p in old_parameters)+sum(p.numel() for p in rma.parameters()),
        training_queries=len(training_dataset), validation_queries=len(validation_dataset),
        batching='Trailing singleton merged; no query dropped; BN sees actual microbatch, not accumulation size.'))
    steps, episodes, history, validation_rows = [], [], [], []
    best_score, best_epoch = -math.inf, None
    rng = random.Random(10)
    for epoch in range(1, args.epochs+1):
        for module in (*adapters, rma):
            module.train()
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        epoch_rows = []
        for step, items in enumerate(tqdm(loader, desc=f'C joint epoch {epoch}/{args.epochs}')):
            shot = (1, 2, 4)[len(steps) % 3]
            rows = []
            for path, category in zip(items['img_path'], items['cls_name']):
                chosen = rng.sample(split['training_support_pool'][category], shot)
                rows.extend(chosen)
                episodes.append(dict(epoch=epoch, step=step+1, query=path, category=category, shot=shot,
                                     references=' | '.join(r['img_path'] for r in chosen)))
            supports = RecordedPromptDataset(str(args.train_data_path), rows, split['classes'],
                                               validation_dataset.transform, validation_dataset.target_transform)
            with torch.no_grad():
                prompt_images = torch.stack([supports[i]['img'] for i in range(len(supports))]).to(device)
                pf, pp = model.encode_image(prompt_images, config.features_list, DPAM_layer=dpam)
                batch = len(items['img_path'])
                pf, pp = pf.reshape(batch, shot, -1), [p.reshape(batch, shot, *p.shape[1:]) for p in pp]
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
            # Must be OUTSIDE no_grad: TA prompt inputs require text-encoder backward.
            globals_, pixels, final_raw = joint_forward(model, adapters, rma, features, patches, pf, pp)
            masks = (items['img_mask'][:, 0] > .5).float().to(device)
            loss, parts = joint_losses(globals_, pixels, final_raw, items['anomaly'].to(device), masks,
                                       focal, dice, args.fusion_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint training loss.')
            group_start = step//args.accumulation*args.accumulation
            group_size = min(args.accumulation, len(loader)-group_start)
            (loss/group_size).backward()
            if (step+1) % args.accumulation == 0 or step+1 == len(loader):
                require_joint_gradients(adapters, rma)
                torch.nn.utils.clip_grad_norm_([*old_parameters, *rma.parameters()], 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            row = dict(epoch=epoch, step=step+1, shot=shot, batch_size=batch,
                       learning_rate=args.learning_rate, adapter_learning_rate=args.adapter_learning_rate,
                       **{k: float(v.detach()) for k, v in parts.items()})
            steps.append(row)
            epoch_rows.append(row)
            del loss, parts, globals_, pixels, final_raw, prompt_images, features, patches, pf, pp
        verify_joint_optimizer(frozen, adapters, rma, optimizer)
        write_csv(output/'training_steps.csv', list(steps[0]), steps)
        write_csv(output/'training_episodes.csv', list(episodes[0]), episodes)
        score, tables = validation(initialized, adapters, rma, output, epoch)
        for shot, table in tables.items():
            validation_rows.extend(dict(epoch=epoch, shot=shot, **r) for r in table)
        write_csv(output/'validation_metrics.csv', ['epoch', 'shot', 'Category', 'Condition', *FIELDS], validation_rows)
        summary = dict(epoch=epoch, **{k: sum(r[k] for r in epoch_rows)/len(epoch_rows)
            for k in epoch_rows[0] if k not in ('epoch', 'step', 'shot', 'batch_size')}, validation_score=score)
        history.append(summary)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        payload = dict(**metadata, **adapter_states(adapters), rma=rma.state_dict(), epoch=epoch,
                       validation_score=score, best_score=best_score, best_epoch=best_epoch)
        if improved:
            atomic_checkpoint(output/'best.pth', payload)
        if args.save_every_epoch:
            atomic_checkpoint(output/f'epoch_{epoch}.pth', payload)
        atomic_checkpoint(output/'last.pth', dict(**payload, optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), support_rng=rng.getstate()))
        write_csv(output/'training_history.csv', list(history[0]), history)
        curves(output, history)
        check_frozen(frozen, optimizer)
        logger.info('epoch=%d global=%.6f local=%.6f fusion=%.6f validation=%.6f best_epoch=%d',
                    epoch, summary['global_loss'], summary['local_loss'], summary['fusion_loss'], score, best_epoch)
        write_json(output/'run_status.json', dict(status='running', complete=False, training_scheme=JOINT,
                   epoch=epoch, best_epoch=best_epoch, best_score=best_score, best_checkpoint=str(output/'best.pth')))
        cleanup()
    return dict(status='completed', complete=True, training_scheme=JOINT, best_epoch=best_epoch,
                best_score=best_score, best_checkpoint=str(output/'best.pth'), epochs=args.epochs,
                frozen_integrity_passed=True, four_adapter_gradient_checks_passed=True)


def main(argv=None):
    args = parse_args(argv)
    if args._output:
        output = project_path(args._output, ROOT)
        if output.parent != (ROOT/'results/train_rma_joint').resolve():
            raise ValueError('Internal output must be a new train_rma_joint session.')
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = new_directory(ROOT/'results'/Path(__file__).stem, 'mvtec_seed10')
    logger = logging.getLogger('rma-joint-training')
    logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(output/'train.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    write_json(output/'experiment_config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(output/'run_status.json', dict(status='running', complete=False, training_scheme=JOINT))
    try:
        write_json(output/'run_status.json', train(args, output, logger))
        print(f'C training completed: {output / "best.pth"}', flush=True)
    except BaseException as error:
        failure_status(output, error)
        logger.exception('Joint training stopped.')
        raise
    finally:
        cleanup()
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
