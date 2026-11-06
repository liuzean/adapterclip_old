"""固定新D第14轮和seed10，只比较RMA局部固定比例，不训练。

先MVTec验证1/2/4-shot，选择0/12.5/25/37.5%之一（25%优先持平）。
若仍选择25%，停止，不重复目标测试；否则统一比例测试六数据集1/2/4-shot。
直接对照G0_L1，同时报告原A和完整新D；每条件独立子进程，失败停止。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rma_fixed_fusion_utils as experiment
from context_alignment import DATASETS
from fusion_ablation_utils import new_directory, project_path, read_json, write_json
from new_d_without_rma import reference_checkpoint
from rma_reliability import child_command, parse_args, target_inputs, target_worker
from rma_reliability_calibration import worker as source_worker
from rma_reliability_utils import source_inputs, summarize
from rma_utils import cleanup, digest, failure_status, run_child


def command(args, session, phase, shot, dataset=None):
    cmd = child_command(args, session, phase, shot, dataset)
    cmd[2] = str(Path(__file__).resolve())
    return cmd


def main(argv=None):
    flags = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    flags.add_argument('--evaluate_all_fractions', action='store_true')
    flags.add_argument('--fixed_history', default='results/rma_fixed_fusion/six_datasets_seed10')
    options, remaining = flags.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = parse_args(remaining, description=__doc__+'\n补齐四比例目标消融：--evaluate_all_fractions（默认只新增12.5%的18轮推理）。')
    if options.evaluate_all_fractions:
        from rma_fixed_fusion_grid import main as grid_main
        if args.calibration or args.calibrate_only:
            flags.error('Target grid is separate from source calibration; omit calibration flags.')
        grid_main(args, project_path(options.fixed_history, ROOT)); return
    checkpoint, checkpoint_hash = reference_checkpoint(args)
    if args._session:
        session = project_path(args._session, ROOT)
        session.relative_to((ROOT/'results'/Path(__file__).stem).resolve())
        campaign = read_json(session/'campaign_config.json')
        if campaign['checkpoint_sha256'] != checkpoint_hash or campaign['implementation_sha256'] != experiment.implementation_hashes():
            raise ValueError('Checkpoint/implementation changed since scheduling.')
        if args._phase == 'calibrate':
            source_worker(args, session, checkpoint, checkpoint_hash, experiment=experiment)
        else:
            target_worker(args, session, checkpoint, checkpoint_hash, experiment=experiment)
        return
    session = new_directory(ROOT/'results'/Path(__file__).stem, 'six_datasets_seed10')
    campaign = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, weight_epoch=14, seed=10,
        rule=experiment.RULE, choices=experiment.CHOICES, selection=experiment.SELECTION,
        implementation_sha256=experiment.implementation_hashes(), source_audit=source_inputs(checkpoint)[2],
        datasets=args.datasets, shots=args.shots, conditions=experiment.CONDITIONS, primary_comparator='G0_L1',
        target_labels_used_for_selection=False, num_workers=args.num_workers, save_maps=args.save_maps,
        save_visualizations=args.save_visualizations,
        history={n: str(getattr(args, n)) for n in ('new_d_run', 'off_run', 'ablation_run',
            'history_session', 'visa_source_root', 'visa_control_root')})
    write_json(session/'campaign_config.json', campaign)
    done = []; source_done = []; phase = 'calibration'; current = None
    try:
        if args.calibration:
            calibration = args.calibration
        else:
            calibration = session/'calibration'; calibration.mkdir()
            for shot in (1, 2, 4):
                current = ['MVTec_validation', shot]
                write_json(session/'run_status.json', dict(status='calibrating', complete=False, phase=phase,
                    current=current, completed_source_shots=source_done, completed_jobs=done))
                print(f'[源域 {len(source_done)+1}/3] MVTec {shot}-shot，固定比例校准（不训练）', flush=True)
                run_child(command(args, session, 'calibrate', shot)); source_done.append(shot); cleanup()
            experiment.lock(calibration, checkpoint, checkpoint_hash)
        locked = experiment.verify_calibration(calibration, checkpoint, checkpoint_hash)
        fraction = locked['rma_fraction']
        campaign.update(calibration=str(calibration), calibration_sha256=digest(calibration/'locked_config.json'), rma_fraction=fraction)
        write_json(session/'campaign_config.json', campaign)
        print(f'源域已锁定统一RMA比例：{fraction:.1%}', flush=True)
        if fraction == .25 or args.calibrate_only:
            reason = 'source_selected_existing_25_percent' if fraction == .25 else 'calibrate_only_requested'
            write_json(session/'run_status.json', dict(status='source_calibration_completed', complete=True,
                target_evaluation_complete=False, target_evaluation_skipped=True, reason=reason,
                calibration=str(calibration), rma_fraction=fraction, completed_source_shots=source_done, completed_jobs=[]))
            print(f'停止于源域阶段：{reason}。没有新增目标集结果，不标为目标验证通过。记录：{session}', flush=True)
            return
        phase = 'target_preflight'; current = None
        jobs = [(d, s) for d in args.datasets for s in args.shots]; audits = {}
        for dataset, shot in jobs:
            _, _, audit = target_inputs(args, dataset, shot, checkpoint_hash)
            audits.setdefault(dataset, {})[str(shot)] = audit
        write_json(session/'input_audit.json', audits)
        phase = 'target'
        for dataset, shot in jobs:
            current = [dataset, shot]
            write_json(session/'run_status.json', dict(status='testing', complete=False, phase=phase, current=current,
                completed_source_shots=source_done, completed_jobs=done, calibration=str(calibration)))
            print(f'[目标 {len(done)+1}/{len(jobs)}] {dataset} {shot}-shot / RMA比例{fraction:.1%}', flush=True)
            run_child(command(args, session, 'target', shot, dataset))
            state = read_json(session/dataset/f'{shot}shot_seed10/run_status.json')
            if not all(state.get(k) is True for k in ('complete', 'endpoint_parity_passed', 'frozen_integrity_passed')):
                raise ValueError('Target round did not pass all checks.')
            done.append((dataset, shot)); summarize(session, done, DATASETS, candidate=experiment.CANDIDATE); cleanup()
        write_json(session/'run_status.json', dict(status='completed', complete=True, target_evaluation_complete=True,
            completed_jobs=done, calibration=str(calibration), rma_fraction=fraction))
    except BaseException as error:
        path = session/'run_status.json'; state = read_json(path) if path.exists() else {}
        state.update(phase=phase, current=current, completed_source_shots=source_done, completed_jobs=done)
        write_json(path, state); failure_status(session, error)
        print(f'已停止：阶段={phase}，当前={current}，源域已完成={source_done}，目标已完成={len(done)}；记录：{session}', flush=True)
        raise
    print(f'Completed: {session}', flush=True)


if __name__ == '__main__':
    main()
