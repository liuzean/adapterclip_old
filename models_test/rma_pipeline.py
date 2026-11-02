"""Train RMA, wait for training exit, then test best.pth on six datasets."""
import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import new_directory, project_path, read_json, write_json
from rma_utils import failure_status, run_child
from test_rma import preflight
from train_rma import parse_args as parse_training_args


def parse_args(argv=None):
    """Pipeline-only arguments plus unchanged train_rma arguments."""
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4), default=[1, 2, 4])
    p.add_argument('--history_session', default='multi_dataset_seed10')
    p.add_argument('--visa_source_root', default='results/branch_diagnostics')
    p.add_argument('--visa_control_root', default='results/dual_output/visa_seed10')
    p.add_argument('--save_maps', action='store_true')
    p.add_argument('--save_visualizations', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--train_help', action='store_true', help='Show train_rma training options and exit.')
    args, training_options = p.parse_known_args(argv)
    if args.train_help:
        parse_training_args(['--help'])
    if len(set(args.datasets)) != len(args.datasets) or args.shots != sorted(set(args.shots)):
        p.error('Unique datasets and ascending unique shots required.')
    if Path(args.history_session).name != args.history_session or args.history_session in ('.', '..'):
        p.error('history_session must be one directory name.')
    training = parse_training_args(training_options)
    if training._output:
        p.error('Internal --_output cannot be supplied to the pipeline.')
    for name in ('visa_source_root', 'visa_control_root'):
        setattr(args, name, project_path(getattr(args, name), ROOT))
    return args, training, training_options


def main(argv=None):
    args, training, training_options = parse_args(argv)
    jobs, audit = preflight(args)
    # Input/path validation before expensive training; no target predictions here.
    for path in (training.base_checkpoint, training.pretrained_model, training.train_data_path/'meta.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    session = new_directory(ROOT/'results'/Path(__file__).stem, 'six_datasets_seed10')
    token = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    train_output = ROOT/'results/train_rma'/f'pipeline_{token}'
    test_output = ROOT/'results/test_rma'/f'pipeline_{token}'
    write_json(session/'input_audit.json', audit)
    write_json(session/'execution_plan.json', dict(train_output=str(train_output), test_output=str(test_output),
        training_arguments=training_options, evaluation_jobs=jobs,
        checkpoint_selection='best full-model MVTec source validation; no target selection'))
    try:
        write_json(session/'run_status.json', dict(status='training', complete=False))
        run_child([sys.executable, '-u', str(ROOT/'models_test/train_rma.py'),
                   *training_options, '--_output', str(train_output)])
        status = read_json(train_output/'run_status.json')
        if status.get('complete') is not True:
            raise RuntimeError('Training did not complete; target testing stopped.')
        best = Path(status['best_checkpoint'])
        if not best.is_file() or best.parent != train_output:
            raise ValueError('Completed training did not supply its own best checkpoint.')
        _, fresh_audit = preflight(args)
        if fresh_audit != audit:
            raise ValueError('Target historical inputs changed during training.')
        write_json(session/'run_status.json', dict(status='testing', complete=False, best_checkpoint=str(best)))
        command = [sys.executable, '-u', str(ROOT/'models_test/test_rma.py'), '--checkpoint', str(best),
                   '--_output', str(test_output), '--datasets', *args.datasets,
                   '--shots', *[str(s) for s in args.shots], '--num_workers', str(training.num_workers)]
        for name in ('history_session', 'visa_source_root', 'visa_control_root'):
            command.extend([f'--{name}', str(getattr(args, name))])
        if args.save_maps:
            command.append('--save_maps')
        if not args.save_visualizations:
            command.append('--no-save_visualizations')
        run_child(command)
        if read_json(test_output/'run_status.json').get('complete') is not True:
            raise RuntimeError('Target evaluation incomplete.')
        write_json(session/'run_status.json', dict(status='completed', complete=True,
            train_output=str(train_output), test_output=str(test_output), best_checkpoint=str(best)))
    except BaseException as error:
        failure_status(session, error)
        raise
    print(f'Pipeline completed. Train: {train_output}\nTest: {test_output}\nManifest: {session}', flush=True)


if __name__ == '__main__':
    main()
