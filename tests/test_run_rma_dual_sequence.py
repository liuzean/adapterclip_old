"""Reject incomplete/wrong-mode dual results before advancing; not auto-run."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
from run_rma_dual_sequence import STAGES, require_complete
from run_rma_followup_sequence import EXPECTED_JOBS


def test_stage_order_and_both_dual_parity_gates():
    assert [r[2] for r in STAGES] == ['rma_fixed_dual_output.py', 'rma_beta4_dual_output.py']
    state = dict(complete=True, target_evaluation_complete=True, completed_jobs=list(EXPECTED_JOBS), mode='fixed125',
        dual_output_parity_passed=True, offline_combination_parity_passed=True)
    require_complete('fixed125', dict(child_status=state))
    with pytest.raises(ValueError):
        require_complete('beta4', dict(child_status=state))
    state['offline_combination_parity_passed'] = False
    with pytest.raises(ValueError):
        require_complete('fixed125', dict(child_status=state))


def test_partial_conditions_cannot_advance_even_when_exit_status_says_complete():
    state = dict(complete=True, target_evaluation_complete=True, completed_jobs=[['Visa', 1]], mode='fixed125',
        dual_output_parity_passed=True, offline_combination_parity_passed=True)
    with pytest.raises(ValueError):
        require_complete('fixed125', dict(child_status=state))
