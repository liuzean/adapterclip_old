"""Manual regression checks for output routing/parity; not run by Codex.

No real datasets, weights or model initialization. These tests deliberately
separate image scores from pixel-map maxima and cover parity failure handling.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))

import dual_output as experiment
from dual_output_utils import (
    FIELDS, SCORE_FIELDS, CategoryBuffer, compare_ablation, compare_scores, route_outputs,
)


class RoutingTests(unittest.TestCase):
    def test_independent_outputs_share_original_tensors_without_modification(self):
        import torch
        maps = {"baseline": torch.full((2, 3, 3), .2), "no_TA": torch.full((2, 3, 3), .8)}
        scores = {"baseline": torch.tensor([.1, .3]), "no_TA": torch.tensor([.6, .9])}
        original_maps = {name: value.clone() for name, value in maps.items()}
        original_scores = {name: value.clone() for name, value in scores.items()}
        actual_maps, actual_scores = route_outputs(maps, scores)
        self.assertIs(actual_maps["dual_output"], maps["no_TA"])
        self.assertIs(actual_scores["dual_output"], scores["baseline"])
        # In particular do not derive the image score from the no_TA map max.
        self.assertFalse(torch.equal(actual_scores["dual_output"], scores["no_TA"]))
        for name in maps:
            torch.testing.assert_close(maps[name], original_maps[name], rtol=0, atol=0)
            torch.testing.assert_close(scores[name], original_scores[name], rtol=0, atol=0)

    def test_disable_restores_both_baseline_outputs(self):
        maps, scores = {"baseline": object(), "no_TA": object()}, {"baseline": object(), "no_TA": object()}
        actual_maps, actual_scores = route_outputs(maps, scores, disable=True)
        self.assertIs(actual_maps["dual_output"], maps["baseline"])
        self.assertIs(actual_scores["dual_output"], scores["baseline"])

    def test_metrics_follow_the_evaluated_output_source_with_two_unique_maps(self):
        import torch
        for disable in (False, True):
            with self.subTest(disable=disable):
                buffer = CategoryBuffer("toy", disable)
                maps = {"baseline": torch.full((2, 2, 2), .2), "no_TA": torch.full((2, 2, 2), .8)}
                scores = {"baseline": torch.tensor([.1, .3]), "no_TA": torch.tensor([.6, .9])}
                mask = torch.tensor([[0, 0], [1, 1]])
                for index in range(2):
                    buffer.append(mask, index, maps, scores, index)
                self.assertEqual(set(buffer.maps), {"baseline", "no_TA"})
                # A class boundary must not retain storage for the whole batch.
                self.assertNotEqual(buffer.maps["baseline"][0].data_ptr(), maps["baseline"][0].data_ptr())

                def metric_fixture(evaluator, category, masks_, maps_, labels, scores_, device, level):
                    value = float(maps_.mean() if level == "pixel" else scores_.mean())
                    return {metric: value for metric in ("AUROC", "AP", "F1max")}

                with patch("branch_diagnostics_utils.evaluate_scores", side_effect=metric_fixture) as evaluate, \
                        patch("tools.Evaluator"):
                    pixels, images = buffer.finish("cpu", MagicMock())
                self.assertEqual(evaluate.call_count, 4)
                pixel_by_name = {row["Condition"]: row for row in pixels}
                image_by_name = {row["Condition"]: row for row in images}
                for metric in ("AUROC", "AP", "F1max"):
                    self.assertEqual(image_by_name["dual_output"][metric], image_by_name["baseline"][metric])
                    self.assertEqual(pixel_by_name["dual_output"][metric], pixel_by_name["baseline" if disable else "no_TA"][metric])
                self.assertFalse(buffer.masks)
                self.assertTrue(all(not value for value in buffer.maps.values()))


class ParityTests(unittest.TestCase):
    def test_score_parity_rejects_nonfinite_or_changed_components(self):
        saved = {field: ".25" for field in SCORE_FIELDS}
        actual = {"Category": "toy", "Image Path": "unused", **{field: .25 for field in SCORE_FIELDS}}
        self.assertTrue(compare_scores(actual, saved, 1e-6, 1e-5)["Passed"])
        for changed in (.4, float("nan"), float("inf")):
            check = compare_scores({**actual, "baseline Image Score": changed}, saved, 1e-6, 1e-5)
            self.assertFalse(check["Passed"])
            self.assertIn("baseline Image Score", check["Failed Fields"])

    def test_full_precision_metrics_checked_for_both_conditions(self):
        old = {(category, name): {field: "80" for field in FIELDS}
               for category in ("toy", "Mean") for name in ("baseline", "no_TA")}
        actual = [{"Category": c, "Condition": n, **{field: 80. for field in FIELDS}} for c, n in old]
        self.assertTrue(all(row["Passed"] for row in compare_ablation({"metrics": old}, actual, .001)))
        actual[-1]["P-AP"] = 81.
        failed = [row for row in compare_ablation({"metrics": old}, actual, .001) if not row["Passed"]]
        self.assertEqual([(row["Condition"], row["Metric"]) for row in failed], [("no_TA", "P-AP")])

    def test_failed_worker_stops_following_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            child = MagicMock()
            child.__enter__.return_value = child
            child.wait.return_value = 1
            with patch.object(experiment, "audit_source", return_value={}), \
                    patch.object(experiment, "audit_sequence", return_value={}), \
                    patch.object(experiment, "select_samples", return_value={}), \
                    patch.object(experiment, "audit_ablation", return_value={"directory": "unused", "hashes": {}}), \
                    patch.object(experiment, "new_directory", return_value=Path(directory)), \
                    patch.object(experiment.subprocess, "Popen", return_value=child) as launch, \
                    patch.object(experiment, "aggregate") as aggregate:
                with self.assertRaisesRegex(RuntimeError, "remaining shots stopped"):
                    experiment.main([])
            self.assertEqual(launch.call_count, 1)
            aggregate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
