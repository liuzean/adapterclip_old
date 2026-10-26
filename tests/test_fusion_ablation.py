"""Optional regression tests for the NEW experiment; no model weights required.

These tests are delivered for manual use. They do not reset Git, undo code or
serve as evidence of model performance. Do not execute without user authorization.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))

import fusion_ablation as experiment
from fusion_ablation_utils import compare_baseline, new_directory


class FusionScoreRegressionTests(unittest.TestCase):
    def test_local_ablation_keeps_three_native_globals_and_original_baseline(self):
        """Dropping local branches must not silently drop their image globals."""
        import torch
        from adaptcliplib import fusion_fun
        from branch_diagnostics_utils import smooth_maps

        batch, side = 2, 4
        features = torch.zeros(batch, 3)
        va_logits = torch.tensor([[0., 1.], [2., 0.]])
        ta_logits = torch.tensor([[1., 0.], [0., 2.]])
        pq_logits = [torch.tensor([[0., 2.], [1., 0.]])] * 4
        local = lambda value: torch.stack((torch.full((batch, side, side), 1 - value),
                                           torch.full((batch, side, side), value)), dim=1)
        va_map, ta_map = local(.2), local(.8)
        pq_maps = [local(value) for value in (.1, .3, .5, .7)]
        align_maps = [torch.full((batch, 1, side, side), value) for value in (.2, .3, .4, .5)]
        visual = MagicMock(return_value=(va_logits, va_map))
        textual = MagicMock()
        textual.static_text_features = torch.zeros(1)
        textual.compute_global_local_score.return_value = (ta_logits, ta_map)
        pq = MagicMock(return_value=(pq_logits, pq_maps, align_maps))
        config = SimpleNamespace(features_list=[6, 12, 18, 24], sigma=1)
        raw, evaluated, scores, maxima, globals_, fixed, extras = experiment.infer_batch(
            config, features, [], features, [], textual, visual, pq, None, ["a", "b"])
        expected_pqa = torch.concat([value[:, 1:2] for value in pq_maps], dim=1).mean(dim=1)
        align = fusion_fun(align_maps, fusion_type="harmonic_mean")[:, 0]
        original = fusion_fun([va_map[:, 1], ta_map[:, 1], expected_pqa], fusion_type="average_mean")
        original = fusion_fun([original, align], fusion_type="harmonic_mean")
        original_global = fusion_fun([va_logits.softmax(-1)[:, 1], ta_logits.softmax(-1)[:, 1],
                                     torch.concat([v.softmax(-1).unsqueeze(-1) for v in pq_logits],
                                                  dim=-1).mean(dim=-1)[:, 1]], fusion_type="average_mean")
        torch.testing.assert_close(raw["baseline"], original)
        torch.testing.assert_close(evaluated["baseline"], smooth_maps(original, 1))
        torch.testing.assert_close(fixed, original_global)
        torch.testing.assert_close(extras["PQA_Fused"], expected_pqa)
        torch.testing.assert_close(raw["no_align"],
                                   (va_map[:, 1] + ta_map[:, 1] + expected_pqa) / 3)
        torch.testing.assert_close(raw["pqa_align"], fusion_fun([expected_pqa, align], fusion_type="harmonic_mean"))
        for condition in experiment.CONDITIONS:
            torch.testing.assert_close(scores[condition], fusion_fun(
                [original_global, maxima[condition]], fusion_type="harmonic_mean"))
        self.assertEqual(set(globals_), {"VA", "TA", "PQA"})
        visual.assert_called_once()
        textual.compute_global_local_score.assert_called_once()
        pq.assert_called_once()

    def test_nonfinite_prediction_fails_instead_of_creating_fake_valid_metrics(self):
        import torch
        config = SimpleNamespace(features_list=[6, 12, 18, 24], sigma=0)
        feature = torch.zeros(1, 3)
        local = torch.full((1, 2, 4, 4), float("nan"))
        logits = torch.zeros(1, 2)
        textual = MagicMock()
        textual.compute_global_local_score.return_value = (logits, local)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            experiment.infer_batch(config, feature, [], feature, [], textual,
                                   MagicMock(return_value=(logits, local)),
                                   MagicMock(return_value=([logits] * 4, [local] * 4,
                                                          [torch.ones(1, 1, 4, 4)] * 4)),
                                   None, ["bad_image"])


class EvidenceAndSchedulingTests(unittest.TestCase):
    def test_baseline_validation_checks_each_metric_including_macro(self):
        fields = [f"{level}-{name}" for level in ("I", "P") for name in experiment.METRICS]
        source = {"metrics": {(category, metric): 50. for category in ("class", "Mean") for metric in fields}}
        rows = [{"Category": category, "Condition": "baseline", **{m: 50. for m in fields}}
                for category in ("class", "Mean")]
        rows[0]["P-AP"] = 49.
        rows[1]["I-AUROC"] = None
        checks = compare_baseline(source, rows, .001)
        self.assertEqual(len(checks), 12)
        self.assertEqual(sum(not row["Passed"] for row in checks), 2)

    def test_repeated_runs_preserve_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            first = new_directory(directory, "run")
            (first / "evidence.txt").write_text("keep", encoding="utf-8")
            second = new_directory(directory, "run")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "evidence.txt").read_text(encoding="utf-8"), "keep")

    def test_failed_worker_stops_later_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            child = MagicMock()
            child.__enter__.return_value = child
            child.wait.return_value = 1
            with patch.object(experiment, "audit_source", return_value={}), \
                    patch.object(experiment, "audit_sequence", return_value={}), \
                    patch.object(experiment, "new_directory", return_value=Path(directory)), \
                    patch.object(experiment.subprocess, "Popen", return_value=child) as launch:
                with self.assertRaisesRegex(RuntimeError, "remaining shots stopped"):
                    experiment.main([])
            self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
