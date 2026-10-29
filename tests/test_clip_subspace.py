"""Manual correctness tests for the CLIP PCA experiment; not run by Codex.

Synthetic mathematical invariants only; no real images, encoders or checkpoints.
These tests validate logic, not algorithm quality or rollback of Git files.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))

from clip_subspace_utils import (
    candidate_outputs, check_reference_separation, compare_controls, fit_subspace,
    patch_descriptors, residual_scores,
)
from dual_output_utils import FIELDS


class SubspaceTests(unittest.TestCase):
    def test_rank_one_support_detects_orthogonal_query(self):
        import torch
        support = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
        state = fit_subspace(support)
        self.assertEqual(state["metadata"]["components"], 1)
        query = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        torch.testing.assert_close(residual_scores(query, state), torch.tensor([0., .25, 0.]))

    def test_constant_support_has_defined_distance_to_mean(self):
        import torch
        state = fit_subspace(torch.tensor([[1., 0.], [1., 0.]]))
        self.assertTrue(state["metadata"]["degenerate"])
        self.assertEqual(state["basis"].shape, (2, 0))
        values = residual_scores(torch.tensor([[1., 0.], [-1., 0.]]), state)
        torch.testing.assert_close(values, torch.tensor([0., 1.]))

    def test_projection_agrees_with_direct_orthogonal_residual(self):
        import torch
        support = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        state = fit_subspace(support, .99)
        query = torch.tensor([[0., 0., 1.], [1., 0., 0.]], dtype=torch.float64)
        centered = query - state["mean"]
        projector = state["basis"] @ state["basis"].T
        expected = (centered @ (torch.eye(3, dtype=torch.float64) - projector)).square().sum(1) / 4
        torch.testing.assert_close(residual_scores(query, state), expected)

    def test_descriptor_excludes_cls_is_batch_independent_and_readonly(self):
        import torch
        layer = torch.tensor([[[99., 99.], [3., 4.]], [[-99., 0.], [4., 3.]]])
        layers = [layer.clone() for _ in range(4)]
        originals = [value.clone() for value in layers]
        descriptors = patch_descriptors(layers)
        torch.testing.assert_close(descriptors, torch.tensor([[[.6, .8]], [[.8, .6]]]))
        torch.testing.assert_close(descriptors[:1], patch_descriptors([value[:1] for value in layers]))
        for value, before in zip(layers, originals):
            torch.testing.assert_close(value, before, rtol=0, atol=0)

    def test_query_order_does_not_change_fit_or_scores(self):
        import torch
        state = fit_subspace(torch.tensor([[1., 0.], [-1., 0.]]))
        before = state["basis"].clone()
        queries = torch.tensor([[1., 0.], [0., 1.]])
        torch.testing.assert_close(residual_scores(queries, state).flip(0), residual_scores(queries.flip(0), state))
        torch.testing.assert_close(state["basis"], before, rtol=0, atol=0)

    def test_mixed_category_batch_uses_its_own_reference_subspace(self):
        import torch
        states = {"x": fit_subspace(torch.tensor([[1., 0.], [-1., 0.]])),
                  "y": fit_subspace(torch.tensor([[0., 1.], [0., -1.]]))}
        features = torch.tensor([[[9., 9.], [1., 0.]], [[9., 9.], [1., 0.]]])
        raw, maps, scores = candidate_outputs([features] * 4, ["x", "y"], states, 2, 0,
                                              None, None, None)
        self.assertEqual(raw.shape, (2, 2, 2))
        torch.testing.assert_close(maps[0], torch.zeros(2, 2))
        torch.testing.assert_close(maps[1], torch.full((2, 2), .25))
        torch.testing.assert_close(scores, torch.tensor([0., .25]))

    def test_reject_invalid_features_and_parameters(self):
        import torch
        support = torch.tensor([[1., 0.], [-1., 0.]])
        for ratio in (0., 1., float("nan")):
            with self.assertRaises(ValueError):
                fit_subspace(support, ratio)
        with self.assertRaises(ValueError):
            fit_subspace(support[:1])
        with self.assertRaises(ValueError):
            patch_descriptors([torch.zeros(1, 2, 2)] * 4)
        with self.assertRaises(ValueError):
            patch_descriptors([torch.ones(1, 3, 2)] * 4)
        invalid = support.clone()
        invalid[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            fit_subspace(invalid)


class ProtocolTests(unittest.TestCase):
    def test_reference_test_overlap_rejected(self):
        source = {"manifest": [{"Image Path": str(ROOT / "query.jpg")}],
                  "reference_paths": [("toy", str(ROOT / "query.jpg"))]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_reference_separation(source)

    def test_disabled_candidate_shares_baseline_without_needing_features_or_fit(self):
        raw, maps, scores = object(), object(), object()
        actual = candidate_outputs(None, None, None, 4, 0, raw, maps, scores, disable=True)
        for value, expected in zip(actual, (raw, maps, scores)):
            self.assertIs(value, expected)

    def test_control_metrics_select_correct_image_and_pixel_source(self):
        old = {("toy", "baseline"): {field: 80. for field in FIELDS},
               ("toy", "no_TA"): {field: 70. for field in FIELDS}}
        actual = [{"Category": "toy", "Condition": "dual_output",
                   **{field: 70. if field.startswith("P-") else 80. for field in FIELDS}}]
        self.assertTrue(all(row["Passed"] for row in compare_controls({"metrics": old}, actual, .001)))
        actual[0]["I-AUROC"] = 70.
        self.assertFalse(all(row["Passed"] for row in compare_controls({"metrics": old}, actual, .001)))


if __name__ == "__main__":
    unittest.main()
