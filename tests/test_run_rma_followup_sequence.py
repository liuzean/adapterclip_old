"""Sequence stops on failure and rejects partial completion; not auto-run."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
import run_rma_followup_sequence as sequence


def test_failed_first_stage_never_starts_second_and_preserves_current_round(tmp_path, monkeypatch):
    monkeypatch.setattr(sequence, 'ROOT', tmp_path)
    scripts = tmp_path/'models_test'; scripts.mkdir()
    for _, _, filename, _, _ in sequence.STAGES:
        (scripts/filename).write_text('', encoding='utf-8')
    commands = []
    def fail(command):
        commands.append(command)
        receipt = Path(command[command.index('--_progress_file')+1])
        session = tmp_path/'results/rma_fixed_fusion/all_fractions_seed10'; session.mkdir(parents=True)
        sequence.write_json(receipt, dict(session=str(session)))
        sequence.write_json(session/'run_status.json', dict(status='failed', complete=False, current=['SDD', 2], completed_jobs=[]))
        raise RuntimeError('simulated failure')
    monkeypatch.setattr(sequence, 'run_child', fail)
    with pytest.raises(RuntimeError, match='simulated'):
        sequence.main([])
    assert len(commands) == 1 and '--evaluate_all_fractions' in commands[0]
    output = tmp_path/'results/run_rma_followup_sequence/three_tasks_seed10'
    state = sequence.read_json(output/'run_status.json')
    assert state['current_stage'] == 'fixed_fraction_grid' and state['completed_stages'] == []
    assert state['stop_progress']['child_status']['current'] == ['SDD', 2]
    assert 'SDD' in (output/'stop_summary.txt').read_text(encoding='utf-8')


def test_process_success_alone_does_not_accept_missing_conditions():
    with pytest.raises(ValueError):
        sequence.require_complete('fixed_fraction_grid', dict(child_status=dict(complete=True,
            target_evaluation_complete=True, completed_jobs=[['Visa', 1]])))
    with pytest.raises(ValueError):
        sequence.require_complete('source_local_diagnostics', dict(child_status=dict(complete=True, completed_shots=[1, 2])))
    sequence.require_complete('source_local_diagnostics', dict(child_status=dict(complete=True, completed_shots=[1, 2, 4])))
