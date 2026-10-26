"""Scheduling regression checks; subprocesses are mocked and no model is loaded."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))

import branch_diagnostics as diagnostics


def child_process(wait_results):
    """Represent Popen's context manager and controllable wait/termination calls."""
    process = MagicMock()
    process.__enter__.return_value = process
    process.wait.side_effect = wait_results
    return process


class BranchDiagnosticsSequenceTests(unittest.TestCase):
    def test_defaults_and_explicit_single_shot_context_override(self):
        args = diagnostics.parse_args([])
        self.assertIsNone(args.k_shots)
        self.assertTrue(args.pq_context)
        self.assertEqual(Path(args.checkpoint_path), ROOT / "checkpoint" /
                         "mvtec_three_adapters_ctx_seed10" / "epoch_15.pth")
        explicit = diagnostics.parse_args(["--k_shots", "2", "--no-pq_context"])
        self.assertEqual(explicit.k_shots, 2)
        self.assertFalse(explicit.pq_context)

    @patch("builtins.print")
    def test_rounds_are_sequential_and_preserve_options(self, _print):
        events = []
        processes = []

        def launch(command, **kwargs):
            shots = int(command[-1])
            events.append(("start", shots))
            process = child_process([])

            def wait():
                events.append(("exit", shots))
                return 0

            process.wait.side_effect = wait
            processes.append(process)
            return process

        options = ["--pq_context", "--save_maps", "--num_workers", "0"]
        with patch.object(diagnostics.subprocess, "Popen", side_effect=launch) as popen:
            diagnostics.run_sequence(options)
        self.assertEqual(events, [("start", 1), ("exit", 1), ("start", 2),
                                  ("exit", 2), ("start", 4), ("exit", 4)])
        for call, shots, process in zip(popen.call_args_list, (1, 2, 4), processes):
            self.assertEqual(call.args[0], [sys.executable, str(ROOT / "models_test" /
                             "branch_diagnostics.py"), *options, "--k_shots", str(shots)])
            self.assertEqual(call.kwargs["cwd"], str(ROOT))
            process.wait.assert_called_once_with()
            process.__exit__.assert_called_once()

    @patch("builtins.print")
    def test_failed_round_stops_remaining_rounds(self, _print):
        processes = [child_process([0]), child_process([7])]
        with patch.object(diagnostics.subprocess, "Popen", side_effect=processes) as popen:
            with self.assertRaises(SystemExit) as stopped:
                diagnostics.run_sequence(["--save_maps"])
        self.assertEqual(stopped.exception.code, 7)
        self.assertEqual(popen.call_count, 2)

    @patch("builtins.print")
    def test_interrupt_waits_for_current_child_and_stops(self, _print):
        process = child_process([KeyboardInterrupt(), 0])
        with patch.object(diagnostics.subprocess, "Popen", return_value=process) as popen:
            with self.assertRaises(SystemExit) as stopped:
                diagnostics.run_sequence([])
        self.assertEqual(stopped.exception.code, 130)
        popen.assert_called_once()
        self.assertEqual(process.wait.call_args_list[-1].kwargs, {"timeout": 15})
        process.terminate.assert_not_called()

    @patch("builtins.print")
    def test_interrupt_timeout_terminates_and_reaps_current_child(self, _print):
        process = child_process([KeyboardInterrupt(), subprocess.TimeoutExpired("diagnostics", 15), 0])
        with patch.object(diagnostics.subprocess, "Popen", return_value=process) as popen:
            with self.assertRaises(SystemExit) as stopped:
                diagnostics.run_sequence([])
        self.assertEqual(stopped.exception.code, 130)
        popen.assert_called_once()
        process.terminate.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 3)


if __name__ == "__main__":
    unittest.main()
