"""Manual-use regression tests; not executed by Codex during development."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))

import pqa_layer_aggregation as experiment
from pqa_layer_aggregation_utils import CategoryBuffer, aggregate_layers


class AggregationTests(unittest.TestCase):
    def pattern(self):
        import torch
        return torch.tensor([[0., .1], [.7, 1.]])

    def test_consensus_weighted_but_all_layers_retained_and_input_unchanged(self):
        import torch
        pattern = self.pattern()
        layers = torch.stack([pattern, pattern * .5, pattern * .8, 1 - pattern]).unsqueeze(0)
        before = layers.clone()
        original, weighted, info = aggregate_layers(layers)
        self.assertGreater(float(info["weights"][0, 0]), float(info["weights"][0, 3]))
        self.assertTrue(bool((info["weights"] >= .125 - 1e-12).all()))
        torch.testing.assert_close(info["weights"].sum(1), torch.ones(1, dtype=torch.float64))
        self.assertTrue(bool((weighted >= layers.min(1)[0] - 1e-6).all()))
        self.assertTrue(bool((weighted <= layers.max(1)[0] + 1e-6).all()))
        torch.testing.assert_close(layers, before, rtol=0, atol=0)
        torch.testing.assert_close(original, layers.mean(1), rtol=0, atol=0)

    def test_constant_layers_and_no_positive_consensus_recover_original_mean(self):
        import torch
        constants = torch.tensor([0., .2, .8, 1.]).view(1, 4, 1, 1).expand(1, 4, 4, 4)
        original, weighted, info = aggregate_layers(constants)
        torch.testing.assert_close(weighted, original, rtol=0, atol=0)
        self.assertTrue(bool(info["fallback"].all()))
        pattern = self.pattern()
        layers = torch.stack([pattern, 1 - pattern, torch.zeros_like(pattern), torch.ones_like(pattern)]).unsqueeze(0)
        original, weighted, info = aggregate_layers(layers)
        self.assertTrue(bool(info["fallback"].all()))
        torch.testing.assert_close(weighted, original, rtol=0, atol=0)

    def test_zero_strength_exact_recovery(self):
        import torch
        layers = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4) / 64
        original, weighted, info = aggregate_layers(layers, 0)
        self.assertIs(original, weighted)
        torch.testing.assert_close(original, layers.mean(1), rtol=0, atol=0)
        torch.testing.assert_close(info["weights"], torch.full((1, 4), .25, dtype=torch.float64))

    def test_weights_do_not_depend_on_other_batch_samples(self):
        import torch
        p = self.pattern()
        first = torch.stack([p, p * .5, p * .8, 1 - p]).unsqueeze(0)
        second = torch.stack([1 - p, p, 1 - p, 1 - p]).unsqueeze(0)
        _, expected, a = aggregate_layers(first)
        _, actual, b = aggregate_layers(torch.cat([first, second]))
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(a["weights"][0], b["weights"][0])

    def test_invalid_values_shapes_and_strengths_rejected(self):
        import torch
        with self.assertRaises(ValueError):
            aggregate_layers(torch.zeros(1, 3, 2, 2))
        for strength in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                aggregate_layers(torch.zeros(1, 4, 2, 2), strength)
        with self.assertRaises(ValueError):
            aggregate_layers(torch.full((1, 4, 2, 2), float("inf")))


class IntegrationLogicTests(unittest.TestCase):
    def test_four_conditions_use_correct_independent_metric_sources(self):
        import torch
        buffer = CategoryBuffer("toy")
        maps = {name: torch.full((2, 2, 2), float(index + 1) / 10)
                for index, name in enumerate(("baseline", "no_TA", "weighted", "weighted_no_TA"))}
        scores = {name: torch.tensor([.1, .2]) if name == "baseline" else torch.tensor([.7, .8]) for name in maps}
        for index in range(2):
            buffer.append(torch.tensor([[0, 1], [0, 1]]), index, maps, scores, index)

        def metric_fixture(evaluator, category, masks, maps_, labels, scores_, device, level):
            value = float(scores_.mean() if level == "image" else maps_.mean())
            return {metric: value for metric in ("AUROC", "AP", "F1max")}

        with patch("tools.Evaluator"), patch("branch_diagnostics_utils.evaluate_scores", side_effect=metric_fixture) as evaluate:
            pixels, images = buffer.finish("cpu", MagicMock())
        self.assertEqual(evaluate.call_count, 6)
        px = {row["Condition"]: row["AP"] for row in pixels}
        im = {row["Condition"]: row["AP"] for row in images}
        self.assertAlmostEqual(px["baseline"], .1, places=6)
        self.assertAlmostEqual(px["dual_output"], .2, places=6)
        self.assertAlmostEqual(px["pqa_weighted"], .3, places=6)
        self.assertAlmostEqual(px["pqa_weighted_dual"], .4, places=6)
        self.assertEqual(im["baseline"], im["dual_output"])
        self.assertEqual(im["pqa_weighted"], im["pqa_weighted_dual"])
        self.assertNotEqual(im["baseline"], im["pqa_weighted"])
        self.assertTrue(all(not values for values in buffer.maps.values()))

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
                    patch.object(experiment.subprocess, "Popen", return_value=child) as launch:
                with self.assertRaisesRegex(RuntimeError, "remaining shots stopped"):
                    experiment.main([])
            self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
