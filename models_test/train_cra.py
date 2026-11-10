r"""实验K训练：加载H第14轮，冻结全部旧模块，仅训练PQA差异表示中的CRA。

python .\models_test\train_cra.py
源域查询/参考逐批复用新D15轮episode；仅源域验证选权重，含epoch0零改动候选。
"""
import argparse
import copy
import logging
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cra_utils import FORMAT, SPEC, attach, evaluate, source_plan, verify_plan
from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, read_json, write_csv
from h_inference import load_model
from rma_control_io import write_json, write_progress
from rma_utils import FIELDS, atomic_checkpoint, check_frozen, cleanup, digest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--h_checkpoint', default='results/train_rma_scratch_new/mvtec_seed10/best.pth')
    p.add_argument('--resume_from', help='Resume this CRA training from its last committed complete epoch.')
    p.add_argument('--_progress_file', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    tokens = sys.argv[1:] if argv is None else argv
    if args.resume_from and any(t.startswith('--') and t.split('=', 1)[0] != '--resume_from' for t in tokens):
        p.error('Resume restores settings; use --resume_from alone.')
    args.h_checkpoint = project_path(args.h_checkpoint, ROOT)
    return args


def validate_source(state, hook, loader, split, output, epoch, progress):
    """Held-out MVTec six-metric/three-shot mean using unchanged H formulas."""
    directory = new_directory(output/'validation', f'epoch_{epoch:03d}')
    root = Path(state.config.test_data_path)
    expected = [dict(Category=r['cls_name'], **{'Image Path': str(root/r['img_path'])},
                     **{'Sample Type': 'Anomaly' if int(r['anomaly']) else 'Normal'}) for r in split['validation_queries']]
    full_memory = state.memory
    rows = []
    hook.cra.eval()
    try:
        for shot in (1, 2, 4):
            state.memory = ({k:v[:shot] for k,v in full_memory[0].items()},
                            {k:[v[:shot] for v in values] for k,values in full_memory[1].items()})
            child = directory/f'{shot}shot'; child.mkdir()
            effective = [r for c in split['classes'] for r in
                         [r for r in split['validation_supports'] if r['cls_name'] == c][:shot]]
            write_json(child/'references.json', effective)
            table = evaluate(state, hook, loader, child, expected=expected, zero_init=epoch == 0,
                progress=lambda batch: progress(phase='source_validation', epoch=epoch, validation_shot=shot, validation_batch=batch))
            rows.extend(dict(epoch=epoch, shot=shot, **r) for r in table)
            cleanup()
    finally:
        state.memory = full_memory
    means = [r for r in rows if r['Category'] == 'Mean' and r['Condition'] == 'K_CRA']
    if len(means) != 3: raise ValueError('Incomplete source validation.')
    score = sum(float(r[f]) for r in means for f in FIELDS)/18
    if not math.isfinite(score): raise ValueError('Nonfinite source selection score.')
    baseline = [r for r in rows if r['Condition'] == 'H']
    baseline_file = output/'source_H_metrics.csv'
    if epoch == 0:
        write_csv(baseline_file, list(baseline[0]), baseline)
    else:
        old = {(r['shot'], r['Category']):r for r in read_csv(baseline_file)}
        if any(abs(float(r[f])-float(old[str(r['shot']), r['Category']][f])) > .001 for r in baseline for f in FIELDS):
            raise ValueError('Frozen source H metrics changed during CRA adaptation.')
    write_csv(directory/'metrics.csv', list(rows[0]), rows)
    return score, str(directory)


def train(args, output, plan, source, split, grouped, logger):
    import torch
    from adaptcliplib import BinaryDiceLoss, FocalLoss
    from fusion_ablation_utils import RecordedPromptDataset
    from rma_control_resume import capture_rng, restore_rng
    from rma_joint_utils import joint_forward, joint_losses
    from tqdm import tqdm
    progress_state = dict(epoch=0, batch=0)
    def progress(**values):
        progress_state.update(values)
        write_progress(output/'training_progress.json', progress_state)
    progress(phase='initializing')
    state = load_model(args.h_checkpoint, source, output, 0)
    cra, hook = attach(state)
    settings = plan['training_settings']
    parameters = list(cra.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], betas=tuple(settings['betas']),
                                  weight_decay=settings['weight_decay'])
    for snapshot in state.snapshots: check_frozen(snapshot, optimizer)
    if sum(p.numel() for p in parameters) != 27652:
        raise ValueError('Unexpected CRA parameter budget.')
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in parameters}:
        raise ValueError('Only CRA parameters may enter the optimizer.')
    base_data = state.loader.dataset
    train_data, val_data = copy.copy(base_data), copy.copy(base_data)
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
    try:
        if args.resume_from and (output/'last.pth').is_file():
            last = torch.load(output/'last.pth', map_location='cpu', weights_only=True)
            if (last.get('format') != FORMAT or last.get('campaign_sha256') != campaign_sha
                    or last.get('h_checkpoint_sha256') != plan['h_checkpoint_sha256'] or not last.get('complete_epoch')):
                raise ValueError('CRA last checkpoint does not match this committed training campaign.')
            cra.load_state_dict(last['cra'], strict=True); optimizer.load_state_dict(last['optimizer'])
            history, best_score, best_epoch, best_file = (last[k] for k in ('history', 'best_score', 'best_epoch', 'best_file'))
            if not 0 <= last['epoch'] <= 15 or [int(r['epoch']) for r in history] != list(range(last['epoch']+1)):
                raise ValueError('CRA committed epoch/history coverage is inconsistent.')
            Path(best_file).resolve().relative_to(output)
            if digest(best_file) != last['best_file_sha256'] or digest(output/'source_H_metrics.csv') != last['source_H_sha256']:
                raise ValueError('Committed source-best checkpoint or H control metrics changed.')
            atomic_checkpoint(output/'best.pth', torch.load(best_file, map_location='cpu', weights_only=True))
            restore_rng(last['resume_rng'], rng)
            gradient_checked = last['gradient_checks_passed']; first_epoch = last['epoch']+1
            write_csv(output/'training_history.csv', list(history[0]), history)
            print(f'CRA从完整epoch={last["epoch"]}之后继续；下一轮={first_epoch}。', flush=True)
            del last
        for epoch in range(first_epoch, 16):
            progress(phase='training' if epoch else 'zero_init_validation', epoch=epoch, batch=0)
            work = new_directory(output/'epochs', f'epoch_{epoch:03d}')
            step_rows = []
            cra.train()
            optimizer.zero_grad(set_to_none=True)
            if epoch:
                batches = [grouped[epoch, step] for step in range(1, 691)]
                sampler = [[indices[path_key(r['query'])] for r in batch] for batch in batches]
                loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=0)
                for step, items in enumerate(tqdm(loader, desc=f'CRA epoch{epoch}/15'), 1):
                    progress(phase='training', epoch=epoch, batch=step)
                    episode = batches[step-1]; shot = int(episode[0]['shot'])
                    if [path_key(p) for p in items['img_path']] != [path_key(r['query']) for r in episode]:
                        raise ValueError('CRA training query order differs from the locked new-D episode.')
                    refs = [support_lookup[p] for r in episode for p in r['references'].split(' | ')]
                    support_data = RecordedPromptDataset(str(root), refs, split['classes'], train_data.transform, train_data.target_transform)
                    prompt = torch.stack([support_data[i]['img'] for i in range(len(support_data))]).to(state.device)
                    with torch.no_grad():
                        features, patches = state.model.encode_image(items['img'].to(state.device), state.config.features_list, DPAM_layer=20)
                        pf, pp = state.model.encode_image(prompt, state.config.features_list, DPAM_layer=20)
                        pf, pp = pf.reshape(2, shot, -1), [p.reshape(2, shot, *p.shape[1:]) for p in pp]
                    before = hook.calls
                    with hook.active(True):
                        globals_, pixels, raw = joint_forward(state.model, state.adapters, state.rma, features, patches, pf, pp)
                    if hook.calls-before != 4: raise ValueError('Missing CRA training layer hooks.')
                    loss, parts = joint_losses(globals_, pixels, raw, items['anomaly'].to(state.device),
                        (items['img_mask'][:, 0] > .5).float().to(state.device), focal, dice, settings['fusion_weight'])
                    if not torch.isfinite(loss) or not loss.requires_grad: raise ValueError('Invalid/detached CRA loss.')
                    group_size = min(4, len(loader)-((step-1)//4)*4)
                    (loss/group_size).backward()
                    if step % 4 == 0 or step == len(loader):
                        grads = [p.grad for p in parameters if p.grad is not None]
                        if not grads or not all(torch.isfinite(g).all() for g in grads):
                            raise ValueError('Missing/nonfinite CRA gradients.')
                        gradient_checked = gradient_checked or any(bool(torch.count_nonzero(g)) for g in grads)
                        if not gradient_checked: raise ValueError('CRA has no nonzero training gradient.')
                        torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip'], error_if_nonfinite=True)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    step_rows.append(dict(epoch=epoch, batch=step, shot=shot,
                        **{k:float(v.detach()) for k,v in parts.items()}))
                    if step % 50 == 0: logger.info('epoch=%s batch=%s total_loss=%.6f', epoch, step, float(loss))
                    del loss, parts, globals_, pixels, raw, prompt, features, patches, pf, pp
                write_csv(work/'training_steps.csv', list(step_rows[0]), step_rows)
            score, validation = validate_source(state, hook, val_loader, split, output, epoch, progress)
            for snapshot in state.snapshots: check_frozen(snapshot, optimizer)
            verify_plan(plan)
            payload = dict(format=FORMAT, h_checkpoint_sha256=plan['h_checkpoint_sha256'], cra_spec=SPEC,
                cra={k:v.detach().cpu().clone() for k,v in cra.state_dict().items()}, epoch=epoch,
                validation_score=score, seed=10, campaign_sha256=campaign_sha)
            weight = work/'weights.pth'; atomic_checkpoint(weight, payload)
            if score > best_score:
                best_score, best_epoch, best_file = score, epoch, str(weight)
                atomic_checkpoint(output/'best.pth', payload)
            summary = dict(epoch=epoch, validation_score=score, best_epoch=best_epoch,
                total_loss=sum(r['total_loss'] for r in step_rows)/len(step_rows) if step_rows else None,
                global_loss=sum(r['global_loss'] for r in step_rows)/len(step_rows) if step_rows else None,
                local_loss=sum(r['local_loss'] for r in step_rows)/len(step_rows) if step_rows else None,
                **{f'gate_{i+1}':float(v) for i,v in enumerate(cra.gates.detach().tanh().cpu())},
                validation_directory=validation, checkpoint=str(weight))
            history.append(summary)
            atomic_checkpoint(output/'last.pth', dict(payload, optimizer=optimizer.state_dict(), history=history,
                best_score=best_score, best_epoch=best_epoch, best_file=best_file, best_file_sha256=digest(best_file),
                source_H_sha256=digest(output/'source_H_metrics.csv'), complete_epoch=True,
                gradient_checks_passed=gradient_checked, resume_rng=capture_rng(rng)))
            write_csv(output/'training_history.csv', list(history[0]), history)
            write_json(output/'run_status.json', dict(complete=False, completed_epochs=epoch,
                best_epoch=best_epoch, best_score=best_score, phase='epoch_complete'))
            logger.info('epoch=%s source_score=%.6f best_epoch=%s gates=%s', epoch, score, best_epoch,
                        cra.gates.detach().tanh().cpu().tolist())
            progress(phase='epoch_complete', completed_epochs=epoch)
            cleanup()
        return dict(complete=True, completed_epochs=15, best_epoch=best_epoch, best_score=best_score,
            selected_checkpoint=str(output/'best.pth'), selected_checkpoint_sha256=digest(output/'best.pth'),
            training_history_sha256=digest(output/'training_history.csv'), source_H_sha256=digest(output/'source_H_metrics.csv'),
            frozen_integrity_passed=True, gradient_checks_passed=gradient_checked, zero_init_parity_passed=True,
            no_op_selected=best_epoch == 0)
    finally:
        hook.close(); cleanup()


def main(argv=None):
    args = parse_args(argv)
    receipt = project_path(args._progress_file, ROOT) if args._progress_file else None
    if args.resume_from:
        output = project_path(args.resume_from, ROOT); output.relative_to((ROOT/'results/train_cra').resolve())
        plan = read_json(output/'campaign_config.json'); verify_plan(plan)
        args.h_checkpoint = Path(plan['h_checkpoint'])
        if read_json(output/'run_status.json').get('complete'):
            print('CRA训练已完成：', output); return
        source, split, grouped, fresh = source_plan(args.h_checkpoint)
        if fresh != plan: raise ValueError('Source training campaign changed.')
    else:
        source, split, grouped, plan = source_plan(args.h_checkpoint)
        output = new_directory(ROOT/'results/train_cra', 'mvtec_seed10')
        write_json(output/'campaign_config.json', plan)
        write_json(output/'run_status.json', dict(complete=False, phase='initializing', completed_epochs=0))
        write_json(output/'source_split.json', split)
    if receipt: write_json(receipt, dict(session=str(output), phase='training'))
    logger = logging.getLogger('train_cra'); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/'train.log', encoding='utf-8'); logger.addHandler(handler)
    try:
        state = train(args, output, plan, source, split, grouped, logger)
        write_json(output/'run_status.json', state)
        if receipt: write_json(receipt, dict(session=str(output), phase='completed'))
        print(f'CRA训练完成：{output}\n源域最优epoch={state["best_epoch"]}；权重={state["selected_checkpoint"]}', flush=True)
    except BaseException as error:
        old = read_json(output/'run_status.json')
        write_json(output/'run_status.json', dict(old, complete=False, error=str(error) or 'KeyboardInterrupt'))
        try: progress = read_json(output/'training_progress.json')
        except (OSError, ValueError): progress = {}
        print(f'CRA训练停止：epoch={progress.get("epoch")}，batch={progress.get("batch")}；后续不启动。\n{output}', flush=True)
        raise
    finally:
        handler.close(); logger.removeHandler(handler); cleanup()


if __name__ == '__main__': main()
