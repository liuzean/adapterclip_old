"""Failure-progress reporting contracts. No model or subprocess required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from fusion_ablation_utils import write_csv, write_json
from rma_utils import failure_status, record_pipeline_progress
from run_rma_pipelines import inspect_progress, stop_summary


def test_training_stop_reports_only_recorded_validated_epochs(tmp_path):
    training = tmp_path/'training'
    training.mkdir()
    write_csv(training/'training_history.csv', ['epoch', 'validation_score'],
              [{'epoch': 1, 'validation_score': 70.}, {'epoch': 2, 'validation_score': 72.}])
    write_json(training/'run_status.json', dict(status='running', complete=False, epoch=2))
    failure_status(training, RuntimeError('failure during epoch 3'))
    receipt = tmp_path/'B_progress.json'
    record_pipeline_progress(receipt, 'training', train_output=str(training), epochs=15, evaluation_jobs=[])
    progress = inspect_progress('B', receipt)
    summary = stop_summary(progress, [], tmp_path, KeyboardInterrupt())
    assert progress['last_validated_epoch'] == 2
    assert '2/15' in summary and '当前方案：B' in summary
    assert progress['training_complete'] is False


def test_test_failure_keeps_dataset_shot_and_completed_count(tmp_path):
    testing = tmp_path/'testing'
    first = testing/'Visa/1shot_seed10'
    failed = testing/'Visa/2shot_seed10'
    first.mkdir(parents=True)
    failed.mkdir(parents=True)
    write_json(first/'run_status.json', dict(status='completed', complete=True))
    write_json(failed/'run_status.json', dict(status='failed', complete=False, error='out of memory'))
    write_json(testing/'run_status.json', dict(status='running', complete=False,
               current=['Visa', 2], completed_jobs=[['Visa', 1]]))
    failure_status(testing, RuntimeError('child failed'))
    receipt = tmp_path/'C_progress.json'
    record_pipeline_progress(receipt, 'testing', test_output=str(testing),
                             evaluation_jobs=[['Visa', 1], ['Visa', 2], ['Visa', 4]])
    progress = inspect_progress('C', receipt)
    summary = stop_summary(progress, ['B'], tmp_path, RuntimeError('child failed'))
    assert (progress['dataset'], progress['shot']) == ('Visa', 2)
    assert progress['completed_test_rounds'] == 1
    assert progress['recorded_error'] == 'out of memory'
    assert '已完成方案：B' in summary and '1/3' in summary and '2-shot' in summary


def test_missing_or_partial_receipt_does_not_guess_another_session(tmp_path):
    unrelated = tmp_path/'unrelated'
    unrelated.mkdir()
    write_json(unrelated/'run_status.json', dict(status='completed', complete=True))
    receipt = tmp_path/'missing.json'
    for content in (None, '{"phase":'):
        if content is not None:
            receipt.write_text(content, encoding='utf-8')
        progress = inspect_progress('B', receipt)
        assert progress['phase'] == 'not_reported'
        assert 'session' not in progress and progress['completed_test_rounds'] == 0


def test_completed_worker_is_not_misreported_as_failed_in_summary(tmp_path):
    testing = tmp_path/'testing'
    worker = testing/'BTAD/1shot_seed10'
    worker.mkdir(parents=True)
    write_json(worker/'run_status.json', dict(status='completed', complete=True))
    write_json(testing/'run_status.json', dict(status='failed', complete=False, current=['BTAD', 1],
                                             error='summary file write failed'))
    receipt = tmp_path/'B_progress.json'
    record_pipeline_progress(receipt, 'testing', test_output=str(testing),
                             evaluation_jobs=[['BTAD', 1], ['BTAD', 2]])
    progress = inspect_progress('B', receipt)
    summary = stop_summary(progress, [], tmp_path, RuntimeError('summary failed'))
    assert progress['round_complete'] is True
    assert '可能停在调度/汇总阶段' in summary
