"""Regression tests against executable blocks extracted from the unchanged test.py."""

import ast
import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_test"))
sys.path.insert(0, str(ROOT))

from adaptcliplib import PQAdapter, VisualAdapter, fusion_fun
from branch_diagnostics import compute_branches, parse_args
from branch_diagnostics_utils import (
    BRANCHES, CategoryBuffer, create_run_dir, evaluate_scores, require_finite,
    sample_key, save_comparison, with_macro_means, write_csv,
)
from tools import Evaluator


def original_inference_block():
    """Compile actual test.py adapter/fusion/smoothing statements, avoiding a copied oracle."""
    tree = ast.parse((ROOT / "test.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test")
    loop = next(node for node in function.body if isinstance(node, ast.For) and "test_dataloader" in ast.unparse(node.iter))
    start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.If) and ast.unparse(node.test) == "args.visual_learner")
    end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
               and ast.unparse(node.targets[0]) == "image_anomaly_pred"
               and "nan_to_num" in ast.unparse(node.value))
    return compile(ast.Module(body=loop.body[start:end + 1], type_ignores=[]), "original_test_py_block", "exec")


class BranchDiagnosticsTests(unittest.TestCase):
    def test_original_block_parity_all_shots_context_and_fusions(self):
        block = original_inference_block()
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        # Real VA and PQA exercise Top-1 matching, context/residual, and four-layer aggregation.
        for device in devices:
            torch.manual_seed(17)
            features = torch.randn(2, 8, device=device)
            patches = [torch.randn(2, 17, 8, device=device) for _ in range(4)]
            text = torch.randn(2, 8, device=device)
            logits = torch.randn(2, 2, device=device)
            local = torch.rand(2, 2, 32, 32, device=device)
            textual = SimpleNamespace(compute_global_local_score=lambda *args: (logits.clone(), local.clone()))
            visual = VisualAdapter(32, 8, input_dim=8).to(device).eval()
            for shots in (1, 2, 4):
                prompts = torch.randn(2, shots, 8, device=device)
                prompt_patches = [torch.randn(2, shots, 17, 8, device=device) for _ in range(4)]
                for context in (False, True):
                    pq = PQAdapter(32, 8, input_dim=8, mid_dim=8, context=context).to(device).eval()
                    for fusion in ("average_mean", "harmonic_mean"):
                        with self.subTest(device=device, shots=shots, context=context, fusion=fusion), torch.no_grad():
                            args = SimpleNamespace(visual_learner=True, textual_learner=True, pq_learner=True,
                                                   fusion_type=fusion, sigma=4)
                            env = dict(torch=torch, np=np, gaussian_filter=gaussian_filter, fusion_fun=fusion_fun,
                                       args=args, k_shots=shots, dataset_name="visa", current_batchsize=2,
                                       device=device, query_feats=features, query_patch_feats=patches,
                                       prompt_feats=prompts, prompt_patch_feats=prompt_patches,
                                       textual_learner=textual, visual_learner=visual, pq_learner=pq,
                                       static_text_features=text, learned_text_features=text)
                            exec(block, env)
                            raw, evaluated, scores = compute_branches(args, features, patches, prompts, prompt_patches,
                                                                      textual, visual, pq, text, text)
                            for name, original in (("VA", "local_vl_map"), ("TA", "local_tl_map"),
                                                   ("PQA", "local_pq_map"), ("ALIGN", "align_score")):
                                torch.testing.assert_close(raw[name], env[original].cpu(), rtol=0, atol=0)
                            torch.testing.assert_close(evaluated["FINAL"], env["pixel_anomaly_map"].cpu(), rtol=0, atol=0)
                            torch.testing.assert_close(scores["FINAL"], env["image_anomaly_pred"].cpu(), rtol=0, atol=0)

    def test_evaluator_matches_original_and_undefined_is_na(self):
        masks = torch.tensor([[[0., 0.], [0., 0.]], [[0., 1.], [1., 0.]]]).int()
        maps = torch.tensor([[[.01, .1], [.2, .15]], [[.2, .9], [.8, .1]]])
        labels, scores = torch.tensor([0, 1]), torch.tensor([.1, .9])
        for level, prefix in (("image", "I"), ("pixel", "P")):
            evaluator = Evaluator("cpu", metrics=[f"{prefix}-{name}" for name in ("AUROC", "AP", "F1max")])
            actual = evaluate_scores(evaluator, "candle", masks, maps, labels, scores, "cpu", level)
            original = evaluator.run(dict(cls_names=np.array(["candle"] * 2), gt_masks=masks, pr_masks=maps,
                                         gt_anomalys=labels, pr_anomalys=scores), "candle")
            self.assertEqual(actual, {name: original[f"{prefix}-{name}"] * 100 for name in actual})
        unavailable = evaluate_scores(evaluator, "candle", masks[:1], maps[:1], labels[:1], scores[:1], "cpu", "pixel")
        self.assertTrue(all(value is None for value in unavailable.values()))

    def test_paths_output_collision_and_na(self):
        args = parse_args(["--checkpoint_path", "checkpoint/epoch_15.pth", "--no-save_visualizations"])
        self.assertEqual(Path(args.checkpoint_path), ROOT / "checkpoint" / "epoch_15.pth")
        self.assertFalse(args.save_visualizations)
        self.assertNotEqual(sample_key(ROOT / "a" / "one.png", ROOT), sample_key(ROOT / "b" / "one.png", ROOT))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertNotEqual(create_run_dir(root, "visa"), create_run_dir(root, "visa"))
            write_csv(root / "metrics.csv", ["value"], [{"value": None}, {"value": float("nan")}])
            with (root / "metrics.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{"value": "N/A"}, {"value": "N/A"}])
        with self.assertRaises(ValueError):
            require_finite("raw", torch.tensor([float("nan")]), ["image.png"])

    def test_statistics_are_pooled_raw_pixels_and_rng_unchanged(self):
        buffer = CategoryBuffer("candle")
        for label in (0, 1):
            mask = torch.full((2, 2), label)
            raw = {name: torch.tensor([[0., 1.], [2., 9.]]) + label for name in BRANCHES}
            maps = {name: value / 10 for name, value in raw.items()}
            buffer.append(mask, label, raw, maps, {name: torch.tensor(float(label)) for name in BRANCHES})
        import logging
        state = torch.get_rng_state().clone()
        pixels, images, stats = buffer.finish("cpu", logging.getLogger("test"))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(pixels), 5)
        self.assertEqual(len(images), 9)
        normal = next(row for row in stats if row["Branch"] == "ALIGN" and row["Sample Type"] == "Normal")
        self.assertEqual(normal["Max"], 9)
        self.assertEqual(normal["P50"], 1.5)
        self.assertEqual(normal["P95"], float(np.percentile([0, 1, 2, 9], 95)))
        macro = with_macro_means(pixels, ["Branch"])
        self.assertEqual(len(macro), 10)

    def test_visualization_and_map_saving_preserve_rng_and_raw_values(self):
        raw = {name: np.arange(16, dtype=np.float32).reshape(4, 4) / 16 for name in BRANCHES}
        originals = {name: value.copy() for name, value in raw.items()}
        evaluated = {name: gaussian_filter(value, sigma=1) for name, value in raw.items()}
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "comparison.png"
            save_comparison(output, torch.zeros(3, 4, 4), np.zeros((4, 4)), raw, evaluated)
            self.assertTrue(output.is_file())
            for name in BRANCHES:
                np.save(Path(temp) / f"{name}.npy", raw[name], allow_pickle=False)
                np.testing.assert_array_equal(np.load(Path(temp) / f"{name}.npy"), originals[name])
                np.testing.assert_array_equal(raw[name], originals[name])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        actual_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], actual_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], actual_numpy[1])
        self.assertEqual(numpy_state[2:], actual_numpy[2:])


if __name__ == "__main__":
    unittest.main()
