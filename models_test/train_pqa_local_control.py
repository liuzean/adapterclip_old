r"""实验M训练：plain/cra两臂均仅更新共享的130参数像素分类层。

python .\models_test\train_pqa_local_control.py --arm plain
python .\models_test\train_pqa_local_control.py --arm cra
固定H14、CRA9、MVTec旧划分与逐批参考；含epoch0，仅源域像素指标选优。
"""
import argparse
import copy
import hashlib
import logging
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cra_utils import attach
from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, write_csv
from h_inference import load_model
from pqa_local_control_utils import (ARMS, CRA_WEIGHT, FORMAT, H_WEIGHT, SETTINGS,
                                     select_source, training_plan, verify_plan)
from rma_control_io import write_json, write_progress
from rma_utils import FIELDS, atomic_checkpoint, check_frozen, cleanup, digest, freeze_base

RESULT_ROOT = ROOT/'results'/Path(__file__).stem


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arm', choices=ARMS, default='plain')
    parser.add_argument('--resume_from', help='Continue after last committed complete epoch; replay an incomplete epoch.')
    parser.add_argument('--_progress_file', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    tokens = sys.argv[1:] if argv is None else argv
    if args.resume_from and any(t.startswith('--') and t.split('=',1)[0] not in ('--resume_from','--_progress_file') for t in tokens):
        parser.error('Resume restores the saved arm/settings; use --resume_from alone.')
    return args


def tensor_digest(head):
    hasher = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        hasher.update(name.encode()); hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def validate(state, engine, head, arm, loader, split, output, epoch, progress):
    from pqa_local_control_model import evaluate
    directory = new_directory(output/'validation', f'epoch_{epoch:03d}')
    root = Path(state.config.test_data_path)
    expected = [dict(Category=r['cls_name'], **{'Image Path':str(root/r['img_path'])},
                     **{'Sample Type':'Anomaly' if int(r['anomaly']) else 'Normal'}) for r in split['validation_queries']]
    original_memory, rows = state.memory, []
    try:
        for shot in (1, 2, 4):
            state.memory = ({k:v[:shot] for k,v in original_memory[0].items()},
                            {k:[v[:shot] for v in values] for k,values in original_memory[1].items()})
            child = directory/f'{shot}shot'; child.mkdir()
            refs = [r for c in split['classes'] for r in
                    [r for r in split['validation_supports'] if r['cls_name'] == c][:shot]]
            write_json(child/'references.json', refs)
            table = evaluate(state, engine, {arm:head}, loader, child, expected, epoch0=epoch == 0,
                             progress=lambda batch,count:progress(phase='source_validation', epoch=epoch,
                                                                    validation_shot=shot, validation_batch=batch))
            rows.extend(dict(Shot=shot, **r) for r in table)
            cleanup()
    finally:
        state.memory = original_memory
    controls = [r for r in rows if r['Condition'] == 'unadapted']
    baseline = output/'source_unadapted_metrics.csv'
    if epoch == 0:
        write_csv(baseline, list(controls[0]), controls)
    else:
        old = {(int(r['Shot']),r['Category']):r for r in read_csv(baseline)}
        if any(abs(float(r[f])-float(old[int(r['Shot']),r['Category']][f])) > .001 for r in controls for f in FIELDS):
            raise ValueError('Frozen source control changed during local-head training.')
    write_csv(directory/'metrics.csv', list(rows[0]), rows)
    return select_source(rows), str(directory)


def train(args, output, source, split, grouped, plan, logger):
    import torch
    from adaptcliplib import BinaryDiceLoss, FocalLoss
    from fusion_ablation_utils import RecordedPromptDataset
    from pqa_local_control_model import FrozenPaths, make_classifier
    from rma_control_resume import capture_rng, restore_rng
    from tqdm import tqdm
    progress_state = dict(epoch=0, batch=0, arm=args.arm)
    def progress(**values):
        progress_state.update(values); write_progress(output/'training_progress.json', progress_state)
    progress(phase='initializing')
    state = engine = hook = None
    try:
        state = load_model(H_WEIGHT, source, output, 0)
        cra, hook = attach(state, CRA_WEIGHT)
        cra_snapshot = freeze_base([None, cra])
        engine = FrozenPaths(state, hook)
        head = make_classifier(state.adapters[2])
        initial_sha = tensor_digest(head)
        params = list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=SETTINGS['learning_rate'], betas=tuple(SETTINGS['betas']),
                                      weight_decay=SETTINGS['weight_decay'])
        if sum(p.numel() for p in params) != 130 or {id(p) for p in params} != {
                id(p) for group in optimizer.param_groups for p in group['params']}:
            raise ValueError('Optimizer must contain exactly the copied 130-parameter classifier.')
        snapshots = (*state.snapshots, cra_snapshot)
        for snapshot in snapshots: check_frozen(snapshot, optimizer)
        train_data, val_data = copy.copy(state.loader.dataset), copy.copy(state.loader.dataset)
        train_data.data_all, train_data.length = split['train_queries'], len(split['train_queries'])
        val_data.data_all, val_data.length = split['validation_queries'], len(split['validation_queries'])
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=False, num_workers=0)
        state.memory = ({k:v.cpu() for k,v in state.memory[0].items()},
                        {k:[v.cpu() for v in values] for k,values in state.memory[1].items()})
        root = Path(state.config.test_data_path)
        indices = {path_key(root/r['img_path']):i for i,r in enumerate(train_data.data_all)}
        support_lookup = {r['img_path']:r for pool in split['training_support_pool'].values() for r in pool}
        focal, dice, rng = FocalLoss(), BinaryDiceLoss(), random.Random(10)
        history, best_score, best_epoch, best_file, first_epoch = [], -math.inf, None, None, 0
        gradient_checked = False
        campaign_sha = digest(output/'campaign_config.json')
        if args.resume_from:
            if (output/'last.pth').is_file():
                last = torch.load(output/'last.pth', map_location='cpu', weights_only=True)
                if (last.get('format') != FORMAT or last.get('campaign_sha256') != campaign_sha
                        or last.get('initial_classifier_sha256') != initial_sha or last.get('arm') != args.arm
                        or not last.get('complete_epoch')):
                    raise ValueError('Last checkpoint is not a committed epoch of this arm.')
                head.load_state_dict(last['classifier'], strict=True); optimizer.load_state_dict(last['optimizer'])
                history, best_score, best_epoch, best_file = (last[k] for k in ('history','best_score','best_epoch','best_file'))
                if [int(r['epoch']) for r in history] != list(range(last['epoch']+1)):
                    raise ValueError('Resume history coverage differs.')
                Path(best_file).resolve().relative_to(output)
                if digest(best_file) != last['best_file_sha256'] or digest(output/'source_unadapted_metrics.csv') != last['source_baseline_sha256']:
                    raise ValueError('Committed best checkpoint or source baseline changed.')
                atomic_checkpoint(output/'best.pth', torch.load(best_file, map_location='cpu', weights_only=True))
                restore_rng(last['resume_rng'], rng)
                gradient_checked = last['gradient_checks_passed']; first_epoch = last['epoch']+1
                write_csv(output/'training_history.csv', list(history[0]), history)
                print(f'M-{args.arm}续训：已提交epoch={last["epoch"]}；下一轮={first_epoch}', flush=True)
                del last
            else:
                print('尚无完整epoch检查点，从epoch0重新开始；保留中断文件。', flush=True)
        for epoch in range(first_epoch, 16):
            progress(phase='training' if epoch else 'epoch0_validation', epoch=epoch, batch=0)
            work = new_directory(output/'epochs', f'epoch_{epoch:03d}')
            steps = []; optimizer.zero_grad(set_to_none=True)
            if epoch:
                batches = [grouped[epoch, step] for step in range(1, 691)]
                sampler = [[indices[path_key(r['query'])] for r in episode] for episode in batches]
                loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=0)
                for step, items in enumerate(tqdm(loader, desc=f'M-{args.arm} epoch{epoch}/15'), 1):
                    progress(phase='training', epoch=epoch, batch=step)
                    episode = batches[step-1]; shot = int(episode[0]['shot'])
                    if [path_key(p) for p in items['img_path']] != [path_key(r['query']) for r in episode]:
                        raise ValueError('Training queries differ from locked new-D episode.')
                    refs = [support_lookup[p] for r in episode for p in r['references'].split(' | ')]
                    supports = RecordedPromptDataset(str(root), refs, split['classes'], train_data.transform, train_data.target_transform)
                    prompt = torch.stack([supports[i]['img'] for i in range(len(supports))]).to(state.device)
                    with torch.no_grad():
                        features, patches = state.model.encode_image(items['img'].to(state.device), state.config.features_list, DPAM_layer=20)
                        pf, pp = state.model.encode_image(prompt, state.config.features_list, DPAM_layer=20)
                        pf, pp = pf.reshape(2, shot, -1), [v.reshape(2, shot, *v.shape[1:]) for v in pp]
                        branch = engine.extract(features, patches, pf, pp, arms=(args.arm,))[args.arm]
                    raw = engine.adapted_raw(branch, head)
                    masks = (items['img_mask'][:, 0] > .5).float().to(state.device)
                    probabilities = torch.stack((1-raw, raw), dim=1)
                    lf = focal(probabilities, masks)
                    la = dice(raw.contiguous(), masks); ln = dice((1-raw).contiguous(), 1-masks)
                    loss = lf+la+ln
                    if not torch.isfinite(loss) or not loss.requires_grad:
                        raise ValueError('Invalid or detached local-only loss.')
                    group_size = min(4, len(loader)-((step-1)//4)*4)
                    (loss/group_size).backward()
                    if step % 4 == 0 or step == len(loader):
                        grads = [p.grad for p in params]
                        if any(g is None or not torch.isfinite(g).all() for g in grads):
                            raise ValueError('Missing or nonfinite classifier gradients.')
                        nonzero = any(bool(torch.count_nonzero(g)) for g in grads)
                        gradient_checked = gradient_checked or nonzero
                        if not gradient_checked: raise ValueError('No nonzero classifier gradients.')
                        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    steps.append(dict(epoch=epoch, batch=step, shot=shot, total_loss=float(loss.detach()),
                                      focal=float(lf.detach()), dice_anomaly=float(la.detach()), dice_normal=float(ln.detach())))
                    if step % 50 == 0: logger.info('epoch=%s batch=%s loss=%.6f', epoch, step, float(loss))
                    del branch, raw, loss, lf, la, ln, probabilities, masks, prompt, features, patches, pf, pp
                write_csv(work/'training_steps.csv', list(steps[0]), steps)
            score, validation = validate(state, engine, head, args.arm, val_loader, split, output, epoch, progress)
            for snapshot in snapshots: check_frozen(snapshot, optimizer)
            verify_plan(plan)
            payload = dict(format=FORMAT, arm=args.arm, epoch=epoch, validation_score=score, settings=SETTINGS,
                h_checkpoint_sha256=plan['h_checkpoint_sha256'], cra_checkpoint_sha256=plan['cra_checkpoint_sha256'],
                initial_classifier_sha256=initial_sha, campaign_sha256=campaign_sha,
                classifier={k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
            weight = work/'weights.pth'; atomic_checkpoint(weight, payload)
            if score > best_score:
                best_score, best_epoch, best_file = score, epoch, str(weight)
                atomic_checkpoint(output/'best.pth', payload)
            history.append(dict(epoch=epoch, validation_score=score, best_epoch=best_epoch,
                total_loss=sum(r['total_loss'] for r in steps)/len(steps) if steps else None,
                validation_directory=validation, checkpoint=str(weight)))
            atomic_checkpoint(output/'last.pth', dict(payload, optimizer=optimizer.state_dict(), history=history,
                best_score=best_score, best_epoch=best_epoch, best_file=best_file, best_file_sha256=digest(best_file),
                source_baseline_sha256=digest(output/'source_unadapted_metrics.csv'), complete_epoch=True,
                gradient_checks_passed=gradient_checked, resume_rng=capture_rng(rng)))
            write_csv(output/'training_history.csv', list(history[0]), history)
            write_json(output/'run_status.json', dict(complete=False, completed_epochs=epoch, best_epoch=best_epoch,
                best_score=best_score, phase='epoch_complete', initial_classifier_sha256=initial_sha))
            progress(phase='epoch_complete', completed_epochs=epoch)
            logger.info('epoch=%s source_pixel_score=%.6f best_epoch=%s', epoch, score, best_epoch)
            cleanup()
        return dict(complete=True, arm=args.arm, completed_epochs=15, best_epoch=best_epoch, best_score=best_score,
            initial_classifier_sha256=initial_sha, selected_checkpoint=str(output/'best.pth'),
            selected_checkpoint_sha256=digest(output/'best.pth'), training_history_sha256=digest(output/'training_history.csv'),
            frozen_integrity_passed=True, gradient_checks_passed=gradient_checked, epoch0_identity_passed=True,
            no_op_selected=best_epoch == 0, trainable_parameters=130)
    finally:
        if engine is not None: engine.close()
        if hook is not None: hook.close()
        state = engine = None; cleanup()


def main(argv=None):
    args = parse_args(argv)
    source, split, grouped, fresh = training_plan()
    if args.resume_from:
        output = project_path(args.resume_from, ROOT); output.relative_to(RESULT_ROOT.resolve())
        plan = read_json(output/'campaign_config.json'); args.arm = plan['arm']; verify_plan(plan)
        if {k:v for k,v in plan.items() if k != 'arm'} != fresh: raise ValueError('Source training contract changed.')
        if read_json(output/'run_status.json').get('complete'):
            print('本臂训练已完成：', output); return
    else:
        plan = dict(fresh, arm=args.arm)
        output = new_directory(RESULT_ROOT, f'{args.arm}_seed10')
        write_json(output/'campaign_config.json', plan); write_json(output/'source_split.json', split)
        write_json(output/'run_status.json', dict(complete=False, completed_epochs=0, phase='initializing'))
    if args._progress_file: write_json(project_path(args._progress_file, ROOT), dict(session=str(output), phase='training'))
    logger = logging.getLogger('M_'+args.arm); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'train.log', encoding='utf-8'); logger.addHandler(handler)
    try:
        status = train(args, output, source, split, grouped, plan, logger)
        write_json(output/'run_status.json', status)
        print(f'M-{args.arm}训练完成：{output}\nbest epoch={status["best_epoch"]}', flush=True)
    except BaseException as error:
        status = read_json(output/'run_status.json'); status.update(complete=False, error=str(error) or 'KeyboardInterrupt')
        write_json(output/'run_status.json', status)
        progress = read_json(output/'training_progress.json') if (output/'training_progress.json').is_file() else {}
        print(f'训练停止：arm={args.arm} epoch={progress.get("epoch")} batch={progress.get("batch")}\n{output}', flush=True)
        raise
    finally:
        handler.close(); logger.removeHandler(handler); cleanup()


if __name__ == '__main__': main()
