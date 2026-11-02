"""Train only RMA; frozen CLIP + VA/TA/PQA stay enabled. Run from project root."""
import argparse
import logging
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_ablation_utils import new_directory, project_path, write_csv, write_json
from rma_utils import (
    FIELDS, MetricsBuffer, atomic_checkpoint, check_frozen, cleanup, curves, digest,
    failure_status, freeze_base, fuse_rma, save_tables, source_spec, source_split, validation_score,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train_data_path', default='E:/projects/datasets/mvtec')
    p.add_argument('--base_checkpoint', default='checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth')
    p.add_argument('--pretrained_model', default='E:/projects/FSAD/AdaptCLIP/checkpoint/train_base/ViT-L-14-336px.pt')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=1e-4)
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
    if min(args.epochs, args.batch_size, args.accumulation, args.top_k, args.match_chunk) < 1 or args.num_workers < 0:
        p.error('Epochs, sizes and accumulation must be positive; workers nonnegative.')
    if args.hidden_dim < 8 or args.hidden_dim % 8:
        p.error('hidden_dim must be divisible by eight.')
    values = (args.learning_rate, args.weight_decay, args.validation_fraction, args.fusion_weight)
    if any(not math.isfinite(v) for v in values) or args.learning_rate <= 0 or min(args.weight_decay, args.fusion_weight) < 0:
        p.error('Invalid finite optimizer/loss settings.')
    if not 0 < args.validation_fraction < 1:
        p.error('validation_fraction must be between zero and one.')
    for name in ('train_data_path', 'base_checkpoint', 'pretrained_model'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args


def validation(initialized, adapter, output, epoch):
    """Held-out source queries, fixed nested supports, full deployed fusion."""
    import torch
    from fusion_ablation import infer_batch
    from test import prompt_association
    config, model, textual, visual, pq, text, memory, loader, dpam, device = initialized
    adapter.eval()
    result = {}
    for shot in (1, 2, 4):
        directory = output/'validation'/f'epoch_{epoch:03d}'/f'{shot}shot'
        directory.mkdir(parents=True)
        pixels, images, buffer = [], [], None
        with torch.no_grad():
            for items in loader:
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                pf, pp = prompt_association(*memory, items['cls_name'])
                pf, pp = pf[:, :shot].to(device), [p[:, :shot].to(device) for p in pp]
                old, old_maps, old_scores, _, _, fixed, extra = infer_batch(
                    config, features, patches, pf, pp, textual, visual, pq, text, items['img_path'])
                g, local = adapter(patches, pp)
                _, maps, scores = fuse_rma(old, extra, fixed, g, local, config.sigma)
                all_maps = dict(baseline=old_maps['baseline'], rma=maps)
                all_scores = dict(baseline=old_scores['baseline'], rma=scores)
                for index, category in enumerate(items['cls_name']):
                    if buffer is None or buffer.category != category:
                        if buffer is not None:
                            p, i = buffer.finish(device)
                            pixels.extend(p)
                            images.extend(i)
                        buffer = MetricsBuffer(category)
                    buffer.append(items['img_mask'][index, 0] > .5, items['anomaly'][index], all_maps, all_scores, index)
        if buffer is None:
            raise ValueError('Empty source validation loader.')
        p, i = buffer.finish(device)
        pixels.extend(p)
        images.extend(i)
        result[shot] = save_tables(directory, pixels, images)
        cleanup()
    return validation_score(result), result


def train(args, output, logger):
    import torch
    from torch.nn import functional as F
    from adaptcliplib import BinaryDiceLoss, FocalLoss
    from fusion_ablation import infer_batch, initialize
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
    config, model, textual, visual, pq, text, memory, val_loader, dpam, device = initialized
    # Cached validation supports on host RAM; only current batch moves to CUDA.
    memory = ({k: v.cpu() for k, v in memory[0].items()},
              {k: [v.cpu() for v in values] for k, values in memory[1].items()})
    initialized[6] = memory
    query_dataset = val_loader.dataset
    query_dataset.data_all = split['validation_queries']
    query_dataset.length = len(query_dataset.data_all)
    import copy
    training_dataset = copy.copy(query_dataset)
    training_dataset.data_all = split['train_queries']
    training_dataset.length = len(training_dataset.data_all)
    loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers)
    frozen = freeze_base(initialized)
    adapter = ReferenceMatchingAdapter(hidden_dim=args.hidden_dim, top_k=args.top_k,
        image_size=config.image_size, chunk=args.match_chunk, matching=args.matching).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    check_frozen(frozen, optimizer)
    focal, dice = FocalLoss(), BinaryDiceLoss()
    base_hash, pretrained_hash = digest(args.base_checkpoint), digest(args.pretrained_model)
    split_hash = digest(output/'source_split.json')
    metadata = dict(format='adaptclip-rma-v1', model_spec=adapter.spec, base_protocol=spec['config'],
        base_sha256=base_hash, pretrained_sha256=pretrained_hash, source_split_sha256=split_hash,
        train_options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')},
        seed=10, selection_rule='source validation full-model category/six-metric/1,2,4-shot equal mean')
    write_json(output/'model_manifest.json', dict(**metadata, trainable_parameters=sum(p.numel() for p in adapter.parameters()),
        frozen_modules=['CLIP', 'VA', 'TA', 'PQA'], train_query_count=len(training_dataset),
        validation_query_count=len(query_dataset)))
    history, steps, episode_rows, validation_rows = [], [], [], []
    best_score, best_epoch = -math.inf, None
    rng = random.Random(10)
    for epoch in range(1, args.epochs+1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_rows = []
        for step, items in enumerate(tqdm(loader, desc=f'RMA epoch {epoch}/{args.epochs}')):
            shot = (1, 2, 4)[(len(steps)) % 3]
            rows = []
            for path, category in zip(items['img_path'], items['cls_name']):
                chosen = rng.sample(split['training_support_pool'][category], shot)
                rows.extend(chosen)
                episode_rows.append(dict(epoch=epoch, step=step+1, query=path, category=category, shot=shot,
                                         references=' | '.join(r['img_path'] for r in chosen)))
            supports = RecordedPromptDataset(str(args.train_data_path), rows, split['classes'],
                                              query_dataset.transform, query_dataset.target_transform)
            with torch.no_grad():
                support_images = torch.stack([supports[i]['img'] for i in range(len(supports))]).to(device)
                pf, pp = model.encode_image(support_images, config.features_list, DPAM_layer=dpam)
                batch = len(items['img_path'])
                pf = pf.reshape(batch, shot, -1)
                pp = [p.reshape(batch, shot, *p.shape[1:]) for p in pp]
                features, patches = model.encode_image(items['img'].to(device), config.features_list, DPAM_layer=dpam)
                old, _, _, _, _, fixed, extra = infer_batch(
                    config, features, patches, pf, pp, textual, visual, pq, text, items['img_path'])
            global_logits, local_logits = adapter(patches, pp)
            probability = local_logits.softmax(1)
            gt = (items['img_mask'][:, 0] > .5).float().to(device)
            global_loss = F.cross_entropy(global_logits, items['anomaly'].long().to(device))
            focal_loss = focal(probability, gt)
            dice_anomaly = dice(probability[:, 1].contiguous(), gt)
            dice_normal = dice(probability[:, 0].contiguous(), 1-gt)
            local_loss = focal_loss + dice_anomaly + dice_normal
            final_raw, _ = fuse_rma(old, extra, fixed, global_logits, local_logits)
            fused_probability = torch.stack((1-final_raw, final_raw), dim=1)
            fusion_loss = (focal(fused_probability, gt) + dice(final_raw.contiguous(), gt)
                           + dice((1-final_raw).contiguous(), 1-gt))
            loss = global_loss + local_loss + args.fusion_weight * fusion_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite RMA training loss.')
            group_start = step//args.accumulation*args.accumulation
            group_size = min(args.accumulation, len(loader)-group_start)
            (loss/group_size).backward()
            if (step+1) % args.accumulation == 0 or step+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            row = dict(epoch=epoch, step=step+1, shot=shot, learning_rate=args.learning_rate,
                global_loss=float(global_loss.detach()), local_loss=float(local_loss.detach()),
                focal_loss=float(focal_loss.detach()), dice_anomaly=float(dice_anomaly.detach()),
                dice_normal=float(dice_normal.detach()), fusion_loss=float(fusion_loss.detach()), total_loss=float(loss.detach()))
            steps.append(row)
            epoch_rows.append(row)
            del loss, global_loss, local_loss, focal_loss, dice_anomaly, dice_normal, fusion_loss
            del final_raw, fused_probability, probability, global_logits, local_logits, support_images, features, patches, pp, pf
        check_frozen(frozen, optimizer)
        write_csv(output/'training_steps.csv', list(steps[0]), steps)
        write_csv(output/'training_episodes.csv', list(episode_rows[0]), episode_rows)
        score, val_tables = validation(initialized, adapter, output, epoch)
        for shot, table in val_tables.items():
            validation_rows.extend(dict(epoch=epoch, shot=shot, **r) for r in table)
        write_csv(output/'validation_metrics.csv', ['epoch', 'shot', 'Category', 'Condition', *FIELDS], validation_rows)
        summary = dict(epoch=epoch, **{k: sum(r[k] for r in epoch_rows)/len(epoch_rows)
                       for k in epoch_rows[0] if k not in ('epoch', 'step', 'shot')}, validation_score=score)
        history.append(summary)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        payload = dict(**metadata, rma=adapter.state_dict(), epoch=epoch, validation_score=score,
                       best_epoch=best_epoch, best_score=best_score)
        if improved:
            atomic_checkpoint(output/'best.pth', payload)
        if args.save_every_epoch:
            atomic_checkpoint(output/f'epoch_{epoch}.pth', payload)
        atomic_checkpoint(output/'last.pth', dict(**payload, optimizer=optimizer.state_dict(),
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                          support_rng=rng.getstate()))
        write_csv(output/'training_history.csv', list(history[0]), history)
        curves(output, history)
        check_frozen(frozen, optimizer)
        logger.info('epoch=%d global_loss=%.6f local_loss=%.6f fusion_loss=%.6f validation=%.6f best_epoch=%d',
                    epoch, summary['global_loss'], summary['local_loss'], summary['fusion_loss'], score, best_epoch)
        write_json(output/'run_status.json', dict(status='running', complete=False, epoch=epoch,
                   best_checkpoint=str(output/'best.pth'), best_epoch=best_epoch, best_score=best_score))
        cleanup()
    return dict(status='completed', complete=True, epochs=args.epochs, best_epoch=best_epoch,
                best_score=best_score, best_checkpoint=str(output/'best.pth'), frozen_integrity_passed=True)


def main(argv=None):
    args = parse_args(argv)
    if args._output:
        output = project_path(args._output, ROOT)
        if output.parent != (ROOT/'results/train_rma').resolve():
            raise ValueError('Internal output must be a new direct train_rma session.')
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = new_directory(ROOT/'results'/Path(__file__).stem, 'mvtec_seed10')
    logger = logging.getLogger('rma-training')
    logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(output/'train.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    write_json(output/'experiment_config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(output/'run_status.json', dict(status='running', complete=False))
    try:
        status = train(args, output, logger)
        write_json(output/'run_status.json', status)
        print(f'Training completed. Best: {status["best_checkpoint"]}', flush=True)
    except BaseException as error:
        failure_status(output, error)
        logger.exception('RMA training stopped.')
        raise
    finally:
        cleanup()
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
