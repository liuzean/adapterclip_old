"""Manual-use regression tests; delivered but not executed during development.

Small tensors, a toy shared head and mocked subprocesses only; no real weights.
"""

import copy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))

import branch_ablation as experiment
from branch_ablation_utils import PQALayerObserver, select_samples


def source_fixture():
    """Synthetic saved order, containing no real data paths or model inputs."""
    rows = [{"Category": category, "Sample Type": label,
             "Image Path": str(ROOT / "synthetic" / category / label / f"{index}.png"),
             "Sample Key": f"{label}_{index}"}
            for category in ("a", "b") for label in ("Normal", "Anomaly") for index in range(3)]
    return {"classes": ["a", "b"], "manifest": rows}


def toy_pqa():
    """Match the stable aliasing pattern: four BNs and ONE shared local head."""
    import torch
    from torch import nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.sharebn = nn.ModuleList([nn.Identity() for _ in range(4)])
            head = nn.Conv2d(3, 2, 1)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.copy_(torch.tensor([0., -1000.]))
            self.local_adapter = nn.ModuleList([head] * 4)

        def forward(self, value):
            return [self.local_adapter[index](self.sharebn[index](value + index)) for index in range(4)]
    return Toy().eval()


class SelectionTests(unittest.TestCase):
    def test_selection_is_fixed_in_order_and_same_across_shots(self):
        first, second = source_fixture(), source_fixture()
        original = copy.deepcopy(first["manifest"])
        info = select_samples([first, second], 2)
        self.assertEqual(info["sample_count_per_shot"], 8)
        self.assertEqual(first["diagnostics"], second["diagnostics"])
        self.assertEqual([r["Sample Key"] for r in first["diagnostics"][:4]],
                         ["Normal_0", "Normal_1", "Anomaly_0", "Anomaly_1"])
        self.assertEqual(first["manifest"], original)

    def test_different_shot_identities_or_missing_types_are_rejected(self):
        first, second = source_fixture(), source_fixture()
        second["manifest"][0]["Image Path"] += ".changed"
        with self.assertRaisesRegex(ValueError, "differ across shots"):
            select_samples([first, second], 2)
        missing = source_fixture()
        missing["manifest"] = [r for r in missing["manifest"] if r["Sample Type"] == "Normal"]
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_samples([missing], 2)


class ObserverTests(unittest.TestCase):
    def test_shared_head_captured_once_per_layer_and_does_not_change_output(self):
        import torch
        pq = toy_pqa()
        sample = source_fixture()["manifest"][0]
        value = torch.ones(2, 3, 2, 2)
        with torch.no_grad():
            expected = pq(value)
            observer = PQALayerObserver(pq, [6, 12, 18, 24], 1, {1: sample})
            with observer:
                actual = pq(value)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(len(observer.rows), 16)  # not 4 duplicate hooks x 4 head calls
        for layer in (6, 12, 18, 24):
            rows = [r for r in observer.rows if r["Layer"] == layer]
            self.assertEqual(len(rows), 4)
            probability = next(r for r in rows if r["Stage"] == "local_probability_before_resize")
            difference = next(r for r in rows if r["Stage"] == "local_logit_difference")
            self.assertEqual(probability["Zero Fraction"], 1.)
            self.assertEqual(difference["Mean"], -1000.)
        self.assertTrue(all(not module._forward_hooks for module in pq.sharebn))
        self.assertFalse(pq.local_adapter[0]._forward_hooks)

    def test_hooks_removed_after_exception(self):
        pq = toy_pqa()
        observer = PQALayerObserver(pq, [6, 12, 18, 24], 1, {0: source_fixture()["manifest"][0]})
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with observer:
                raise RuntimeError("interrupted")
        self.assertFalse(observer.handles)
        self.assertTrue(all(not module._forward_hooks for module in pq.sharebn))
        self.assertFalse(pq.local_adapter[0]._forward_hooks)

    def test_missing_head_calls_are_not_reported_as_success(self):
        pq = toy_pqa()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            with PQALayerObserver(pq, [6, 12, 18, 24], 1, {0: source_fixture()["manifest"][0]}):
                pass
        self.assertFalse(pq.local_adapter[0]._forward_hooks)


class PixelAblationTests(unittest.TestCase):
    def test_correct_removed_branch_denominator_fixed_globals_and_single_forward(self):
        import torch
        from adaptcliplib import fusion_fun
        batch = 2
        features = torch.zeros(batch, 3)
        local = lambda probability: torch.stack([torch.full((batch, 4, 4), 1 - probability),
                                                torch.full((batch, 4, 4), probability)], dim=1)
        va, ta = local(.2), local(.8)
        pq_maps = [local(.4)] * 4
        distance_maps = [torch.full((batch, 1, 4, 4), .3)] * 4
        va_global, ta_global, pq_global = torch.tensor([[0., 1.]] * batch), torch.tensor([[1., 0.]] * batch), torch.zeros(batch, 2)
        textual = MagicMock()
        textual.compute_global_local_score.return_value = (ta_global, ta)
        visual = MagicMock(return_value=(va_global, va))
        pq = MagicMock(return_value=([pq_global] * 4, pq_maps, distance_maps))
        raw, evaluated, scores, maxima, globals_, fixed, extras, distances = experiment.infer_batch(
            SimpleNamespace(features_list=[6, 12, 18, 24], sigma=0), features, [], features, [],
            textual, visual, pq, None, ["a", "b"], nullcontext())
        pqa = pq_maps[0][:, 1]
        align = fusion_fun(distance_maps, fusion_type="harmonic_mean")[:, 0]
        for name, inputs in (("baseline", [va[:, 1], ta[:, 1], pqa]),
                             ("no_VA", [ta[:, 1], pqa]), ("no_TA", [va[:, 1], pqa])):
            expected = fusion_fun([fusion_fun(inputs, fusion_type="average_mean"), align], fusion_type="harmonic_mean")
            torch.testing.assert_close(raw[name], expected)
            torch.testing.assert_close(evaluated[name], expected)
            torch.testing.assert_close(scores[name], fusion_fun([fixed, maxima[name]], fusion_type="harmonic_mean"))
        expected_global = fusion_fun([va_global.softmax(-1)[:, 1], ta_global.softmax(-1)[:, 1],
                                     pq_global.softmax(-1)[:, 1]], fusion_type="average_mean")
        torch.testing.assert_close(fixed, expected_global)
        self.assertEqual(set(globals_), {"VA", "TA", "PQA"})
        self.assertEqual(len(distances), 4)
        torch.testing.assert_close(extras["PQA_Fused"], pqa)
        visual.assert_called_once()
        textual.compute_global_local_score.assert_called_once()
        pq.assert_called_once()

    def test_failed_worker_stops_following_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            child = MagicMock()
            child.__enter__.return_value = child
            child.wait.return_value = 1
            with patch.object(experiment, "audit_source", return_value={"diagnostics": []}), \
                    patch.object(experiment, "audit_sequence", return_value={}), \
                    patch.object(experiment, "select_samples", return_value={}), \
                    patch.object(experiment, "new_directory", return_value=Path(directory)), \
                    patch.object(experiment.subprocess, "Popen", return_value=child) as launch:
                with self.assertRaisesRegex(RuntimeError, "remaining shots stopped"):
                    experiment.main([])
            self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
