"""VisA seed=10 fusion ablations with selected PQA four-layer diagnostics.

Run from models_test: python fusion_ablation.py
Default: isolated sequential 1/2/4-shot workers; no training or model edits.
"""

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_ablation_utils import (
    CONDITIONS, METRICS, STAT_FIELDS, RecordedPromptDataset, audit_sequence,
    audit_source, compare_baseline, compute_conditions, map_statistics,
    metric_tables, new_directory, path_key, project_path, read_csv,
    save_panels, write_csv, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/branch_diagnostics")
    parser.add_argument("--k_shots", type=int, choices=(1, 2, 4), default=None,
                        help="Omit to run 1, 2, 4 shots sequentially.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers; 0 limits host RAM, source runs used 4.")
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true",
                        help="Save raw/evaluated maps ONLY for the four fixed diagnostic images.")
    parser.add_argument("--baseline_metric_atol", type=float, default=0.001,
                        help="Allowed absolute metric difference, percentage points.")
    parser.add_argument("--baseline_map_atol", type=float, default=1e-6)
    parser.add_argument("--baseline_map_rtol", type=float, default=1e-5)
    parser.add_argument("--_worker_session", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    import math
    if args.num_workers < 0 or any(not math.isfinite(v) or v < 0 for v in (
            args.baseline_metric_atol, args.baseline_map_atol, args.baseline_map_rtol)):
        parser.error("Worker count and finite tolerances must be nonnegative.")
    if args._worker_session and args.k_shots is None:
        parser.error("Internal worker requires one explicit shot count.")
    args.source_root = project_path(args.source_root, PROJECT_ROOT)
    return args


def initialize(source, output, num_workers, logger):
    """Match the stable setup, replacing only random reference construction.

    Reuse Dataset, transforms, adapters and test.build_prompt_memory. Reference
    batches retain the source batch size and exact recorded order (including
    duplicates); this also preserves batch-sensitive prompt encoding behavior.
    """
    import torch
    import adaptcliplib
    from adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter
    from dataset import Dataset
    from test import build_prompt_memory, resolve_pretrained_model_type
    from tools import get_transform

    config = SimpleNamespace(**source["config"])
    if not torch.cuda.is_available():
        raise RuntimeError("The original static text adapter requires CUDA.")
    device = "cuda"
    model_type = resolve_pretrained_model_type(config.pretrained_model)
    if model_type == "ViT-L/14@336px":
        patch_size, input_dim, dpam = 14, 768, 20
    elif model_type == "VITB16_PLUS_240":
        patch_size, input_dim, dpam = 16, 640, 10
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    if config.image_size % patch_size:
        raise ValueError("image_size must be divisible by patch size.")
    model, _ = adaptcliplib.load(config.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=dpam)
    transform, target_transform = get_transform(image_size=config.image_size)
    prompts = RecordedPromptDataset(config.test_data_path, source["references"],
                                    source["classes"], transform, target_transform)
    queries = Dataset(root=config.test_data_path, transform=transform,
                      target_transform=target_transform, dataset_name=config.dataset,
                      k_shots=config.k_shots, save_dir=str(output), mode="test", seed=config.seed)
    prompt_loader = torch.utils.data.DataLoader(prompts, batch_size=config.batch_size, shuffle=False)
    query_loader = torch.utils.data.DataLoader(queries, batch_size=config.batch_size,
                                              shuffle=False, num_workers=num_workers)
    textual = TextualAdapter(model.to("cpu"), config.image_size, config.n_ctx)
    visual = VisualAdapter(config.image_size, patch_size, input_dim=input_dim, reduction=config.vl_reduction)
    pq = PQAdapter(config.image_size, patch_size, context=config.pq_context,
                   input_dim=input_dim, mid_dim=config.pq_mid_dim, layers_num=len(config.features_list))
    logger.info("Loading existing checkpoint: %s", config.checkpoint_path)
    checkpoint = torch.load(config.checkpoint_path)
    for name, adapter in (("textual_learner", textual), ("visual_learner", visual), ("pq_learner", pq)):
        adapter.load_state_dict(checkpoint[name])
    del checkpoint
    for module in (model, textual, visual, pq):
        module.to(device)
        module.eval()
    textual.prepare_static_text_feature(model)
    with torch.no_grad():
        learned_prompts, tokens = textual()
        learned_text = model.encode_text_learn(learned_prompts, tokens).float()
    memory = build_prompt_memory(model, prompt_loader, device, queries.obj_list,
                                 queries.view_list, config.features_list, dpam)
    return config, model, textual, visual, pq, learned_text, memory, query_loader, dpam, device


def infer_batch(config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths):
    """One adapter pass -> raw maps, smoothed maps, scores and original PQA layers.

    Globals: [B,D]; patch lists: four [B,1+L,D]; prompt patches:
    four [B,S,1+L,D]. PQA returns four local [B,2,H,W] probabilities.
    Select anomaly channel -> four [B,H,W]; their mean is original PQA.
    All image variants use the SAME original mean of three native globals [B].
    Score extraction/global aggregation mirrors test.py and
    branch_diagnostics.compute_branches; PQAdapter.forward itself is not copied.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import require_finite, smooth_maps

    va_global, va_local = visual(features, patches, textual.static_text_features)
    ta_global, ta_local = textual.compute_global_local_score(features, patches, learned_text)
    pq_global, pq_local, align_local = pq(features, patches, prompt_features, prompt_patches)
    if len(pq_local) != 4 or len(align_local) != 4:
        raise ValueError("Original PQA must return four local and alignment maps.")
    layers = {f"PQA_L{layer}": value[:, 1].detach()
              for layer, value in zip(config.features_list, pq_local)}
    pqa = torch.concat([value.unsqueeze(1) for value in layers.values()], dim=1).mean(dim=1).detach()
    align = fusion_fun(align_local, fusion_type="harmonic_mean")[:, 0]
    raw = compute_conditions(va_local[:, 1].detach(), ta_local[:, 1].detach(), pqa, align)
    if isinstance(pq_global, list):
        pq_score = torch.concat([value.softmax(-1).unsqueeze(-1) for value in pq_global],
                                dim=-1).mean(dim=-1).detach()[:, 1].detach()
    else:
        pq_score = pq_global.softmax(-1)[:, 1].detach()
    globals_ = {"VA": va_global.softmax(-1)[:, 1].detach(),
                "TA": ta_global.softmax(-1)[:, 1].detach(), "PQA": pq_score}
    fixed_global = fusion_fun(list(globals_.values()), fusion_type="average_mean")
    evaluated, scores, maxima = {}, {}, {}
    for name in CONDITIONS:
        require_finite(f"{name} raw", raw[name], paths)
        smoothed = smooth_maps(raw[name], config.sigma).to(features.device)
        maximum = torch.max(smoothed.view(features.shape[0], -1), dim=1)[0]
        score = fusion_fun([fixed_global, maximum], fusion_type="harmonic_mean")
        require_finite(f"{name} smoothed", smoothed, paths)
        require_finite(f"{name} image score", score, paths)
        # Same sanitization as stable test.py, after fusion; finite checks fail first.
        evaluated[name] = torch.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0).cpu()
        scores[name] = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).cpu()
        maxima[name] = maximum.cpu()
    for name, value in {**layers, **globals_, "PQA_Fused": pqa, "ALIGN": align,
                        "fixed_global": fixed_global}.items():
        require_finite(name, value, paths)
    return raw, evaluated, scores, maxima, globals_, fixed_global, {**layers, "PQA_Fused": pqa, "ALIGN": align}


class CategoryBuffer:
    """Bound CPU storage to one class: uint8 GT and three evaluated map variants."""

    def __init__(self, category):
        self.category = category
        self.masks, self.labels = [], []
        self.maps = {name: [] for name in CONDITIONS}
        self.scores = {name: [] for name in CONDITIONS}

    def append(self, mask, label, maps, scores, index):
        import torch
        self.masks.append(mask.to(torch.uint8).clone())
        self.labels.append(label)
        for name in CONDITIONS:
            # clone avoids retaining an entire batch when the class changes mid-batch.
            self.maps[name].append(maps[name][index].clone())
            self.scores[name].append(scores[name][index].clone())

    def finish(self, device, logger):
        """Use unchanged evaluator formulas with fresh state for each condition/level."""
        import torch
        from tools import Evaluator
        from branch_diagnostics_utils import evaluate_scores
        masks = torch.stack(self.masks).int()
        labels = torch.tensor(self.labels).int()
        pixels, images = [], []
        for name in CONDITIONS:
            maps = torch.stack(self.maps[name])
            self.maps[name].clear()
            scores = torch.stack(self.scores[name])
            evaluator = Evaluator(device, metrics=[f"P-{metric}" for metric in METRICS])
            pixels.append({"Category": self.category, "Condition": name, **evaluate_scores(
                evaluator, self.category, masks, maps, labels, scores, device, "pixel")})
            del evaluator, maps
            dummy = torch.zeros((len(labels), 1, 1))
            evaluator = Evaluator(device, metrics=[f"I-{metric}" for metric in METRICS])
            images.append({"Category": self.category, "Condition": name, **evaluate_scores(
                evaluator, self.category, dummy, dummy, labels, scores, device, "image")})
            del evaluator, dummy, scores
            logger.info("%s / %s: pixel=%s image=%s", self.category, name, pixels[-1], images[-1])
        self.masks.clear()
        self.labels.clear()
        for values in self.scores.values():
            values.clear()
        return pixels, images


def diagnose_sample(args, source, output, sample, image, mask, raw, extras, index):
    """Inspect only four manifest-selected images; no subset performance claims."""
    import numpy as np
    import torch
    from branch_diagnostics_utils import require_finite, smooth_maps
    maps = {**{name: value[index].detach().cpu() for name, value in extras.items()},
            **{name: value[index].detach().cpu() for name, value in raw.items()}}
    expected_shape = tuple(mask.shape)
    for name, value in maps.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Diagnostic shape mismatch: {name} {value.shape} != {expected_shape}")
        require_finite(name, value, [sample["Image Path"]])
    reconstructed = torch.concat([maps[f"PQA_L{layer}"].unsqueeze(0)
                                  for layer in source["config"]["features_list"]], dim=0).mean(dim=0)
    # Reconstruction is on CPU whereas the original mean is on CUDA; permit only
    # normal reduction rounding. Stored original maps provide an independent check.
    if not torch.allclose(reconstructed, maps["PQA_Fused"], atol=1e-6, rtol=1e-5):
        raise ValueError("Four returned layer maps do not reconstruct the original PQA mean within tolerance.")
    smooth = {name: smooth_maps(value.unsqueeze(0), source["config"]["sigma"])[0]
              for name, value in maps.items()}
    raw_arrays = {name: value.numpy() for name, value in maps.items()}
    evaluated_arrays = {name: value.numpy() for name, value in smooth.items()}
    mask_array = mask.numpy()
    statistics = [map_statistics(source["config"]["k_shots"], sample, name, stage, value, mask_array)
                  for stage, arrays in (("raw", raw_arrays), ("evaluated", evaluated_arrays))
                  for name, value in arrays.items()]
    key = sample["Sample Key"]
    relative = Path(sample["Category"]) / sample["Sample Type"] / key
    if args.save_visualizations:
        title = f"VisA {source['config']['k_shots']}-shot seed=10 / {key}"
        save_panels(output / "visualizations" / "pqa_layers" / relative.with_suffix(".png"),
                    image, mask_array, raw_arrays, evaluated_arrays,
                    [f"PQA_L{layer}" for layer in source["config"]["features_list"]]
                    + ["PQA_Fused", "ALIGN", "baseline"], 3, title)
        save_panels(output / "visualizations" / "fusion_conditions" / relative.with_suffix(".png"),
                    image, mask_array, raw_arrays, evaluated_arrays, list(CONDITIONS), 5, title)
    if args.save_maps:
        for stage, arrays in (("raw", raw_arrays), ("evaluated", evaluated_arrays)):
            directory = output / "raw_maps" / relative / stage
            directory.mkdir(parents=True, exist_ok=True)
            for name, array in arrays.items():
                np.save(directory / f"{name}.npy", array, allow_pickle=False)
    checks = []
    for stage, field, arrays in (("raw", "Raw Maps", raw_arrays),
                                ("evaluated", "Evaluated Maps", evaluated_arrays)):
        for current, original in (("PQA_Fused", "PQA"), ("ALIGN", "ALIGN"), ("baseline", "FINAL")):
            old_dir = sample.get(field)
            path = Path(source["source"]) / old_dir / f"{original}.npy" if old_dir else None
            row = {"Image Path": sample["Image Path"], "Stage": stage, "Map": original,
                   "Reference": str(path) if path else "", "Status": "unavailable", "Max Abs Delta": None}
            if path is not None and path.is_file():
                previous = np.load(path, allow_pickle=False)
                current_array = arrays[current]
                compatible = previous.shape == current_array.shape
                passed = compatible and np.isfinite(previous).all() and np.allclose(
                    previous, current_array, atol=args.baseline_map_atol, rtol=args.baseline_map_rtol)
                row.update({"Status": "passed" if passed else "failed",
                            "Max Abs Delta": float(np.max(np.abs(previous - current_array))) if compatible else None})
            checks.append(row)
    return statistics, checks


def run_inference(args, source, output, logger):
    """One full test pass per shot; emit three variants from shared predictions."""
    import torch
    from tools import setup_seed
    from test import prompt_association
    from tqdm import tqdm
    setup_seed(source["config"]["seed"])
    config, model, textual, visual, pq, learned_text, memory, loader, dpam, device = initialize(
        source, output, args.num_workers, logger)
    selected = {path_key(row["Image Path"]): row for row in source["diagnostics"]}
    visited = set()
    pixels, images, statistics, map_checks, score_rows, category_checks = [], [], [], [], [], []
    buffer, processed, closed = None, 0, set()

    def flush(current):
        p_rows, i_rows = current.finish(device, logger)
        pixels.extend(p_rows)
        images.extend(i_rows)
        combined = metric_tables(output, pixels, images)
        checks = compare_baseline(source, [row for row in combined if row["Category"] == current.category],
                                  args.baseline_metric_atol)
        category_checks.extend(checks)
        write_csv(output / "baseline_reference_comparison.csv", list(category_checks[0]), category_checks)
        write_csv(output / "image_scores.csv", list(score_rows[0]), score_rows)
        if not all(row["Passed"] for row in checks):
            raise RuntimeError(f"Baseline metrics differ for {current.category}; see baseline_reference_comparison.csv. Remaining shots stopped.")
        gc.collect()
        torch.cuda.empty_cache()

    for items in tqdm(loader, desc=f"Fusion ablation {config.k_shots}-shot"):
        paths = items["img_path"]
        with torch.no_grad():
            features, patches = model.encode_image(items["img"].to(device), config.features_list, DPAM_layer=dpam)
            prompt_features, prompt_patches = prompt_association(*memory, items["cls_name"])
            raw, evaluated, scores, maxima, globals_, fixed_global, extras = infer_batch(
                config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths)
        masks = items["img_mask"][:, 0].clone()
        masks[masks > 0.5], masks[masks <= 0.5] = 1, 0
        for name in CONDITIONS:
            if evaluated[name].shape != masks.shape:
                raise ValueError(f"{name}: map {evaluated[name].shape} != GT {masks.shape}")
        for index, category in enumerate(items["cls_name"]):
            label = int(items["anomaly"][index])
            path = paths[index]
            expected = source["manifest"][processed]
            if (path_key(path) != path_key(expected["Image Path"]) or category != expected["Category"]
                    or label != int(expected["Sample Type"] == "Anomaly")):
                raise ValueError(f"Runtime query order/category/label differs from source: {path}")
            if buffer is None or buffer.category != category:
                if buffer is not None:
                    flush(buffer)
                    closed.add(buffer.category)
                if category in closed:
                    raise ValueError("Test classes must be contiguous to release category buffers safely.")
                buffer = CategoryBuffer(category)
            buffer.append(masks[index], label, evaluated, scores, index)
            score_row = {"Category": category, "Image Path": path, "Label": label,
                         **{f"{name} Global": float(value[index]) for name, value in globals_.items()},
                         "Fixed Global Mean": float(fixed_global[index])}
            for name in CONDITIONS:
                score_row.update({f"{name} Map Max": float(maxima[name][index]),
                                  f"{name} Image Score": float(scores[name][index])})
            score_rows.append(score_row)
            if path_key(path) in selected:
                rows, checks = diagnose_sample(args, source, output, selected[path_key(path)],
                                               items["img"][index], masks[index], raw, extras, index)
                statistics.extend(rows)
                map_checks.extend(checks)
                visited.add(path_key(path))
                write_csv(output / "pqa_layer_statistics.csv", STAT_FIELDS,
                          [r for r in statistics if r["Map"].startswith("PQA_L")])
                write_csv(output / "diagnostic_statistics.csv", STAT_FIELDS,
                          [r for r in statistics if not r["Map"].startswith("PQA_L")])
                write_csv(output / "baseline_map_checks.csv", list(map_checks[0]), map_checks)
                if any(row["Status"] == "failed" for row in checks):
                    raise RuntimeError("Stored baseline map mismatch; see baseline_map_checks.csv. Remaining shots stopped.")
            processed += 1
        # No raw maps or four-layer tensors are retained beyond this batch.
        del features, patches, prompt_features, prompt_patches, raw, extras, evaluated, scores, maxima, globals_, fixed_global
    if buffer is None or processed != len(source["manifest"]) or visited != set(selected):
        raise RuntimeError("Incomplete test set or diagnostic sample coverage.")
    flush(buffer)
    combined = metric_tables(output, pixels, images)
    checks = compare_baseline(source, combined, args.baseline_metric_atol)
    write_csv(output / "baseline_reference_comparison.csv", list(checks[0]), checks)
    write_csv(output / "image_scores.csv", list(score_rows[0]), score_rows)
    if not all(row["Passed"] for row in checks):
        raise RuntimeError("Baseline metrics differ from stored FINAL; inspect baseline_reference_comparison.csv before interpreting ablations.")
    write_csv(output / "metrics_combined.csv", ["Category", "Condition", *[
        f"{level}-{metric}" for level in ("I", "P") for metric in METRICS]], combined)
    return {"status": "completed", "complete": True, "sample_count": processed,
            "category_count": len(closed) + 1, "conditions": list(CONDITIONS),
            "diagnostic_sample_count": len(visited), "baseline_metrics": "passed_within_tolerance",
            "pqa_layer_reconstruction": "passed_within_tolerance",
            "baseline_map_checks": {state: sum(row["Status"] == state for row in map_checks)
                                    for state in ("passed", "failed", "unavailable")},
            "fresh_test_py_rerun": False}


def run_worker(args):
    session = Path(args._worker_session).resolve()
    session.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    output = session / f"visa_{args.k_shots}shot_seed10"
    output.mkdir()  # refuse to overwrite/reuse even an incomplete shot directory
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    logger = logging.getLogger("fusion_ablation")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        source = audit_source(args.source_root / f"visa_{args.k_shots}shot_seed10", args.k_shots, PROJECT_ROOT)
        import torch
        import numpy as np
        import scipy
        write_json(output / "experiment_config.json", {
            **source["config"], "source_result_dir": source["source"], "result_dir": str(output),
            "num_workers": args.num_workers, "save_visualizations": args.save_visualizations,
            "save_maps": args.save_maps, "maps_scope": "four_fixed_diagnostic_samples_only",
            "conditions": list(CONDITIONS), "native_global_policy": "Fixed mean(VA,TA,PQA) for all conditions",
            "image_score": "Original harmonic_mean(fixed_global, max(smoothed_condition_map))",
            "metric_scale": "0-100; equal macro across categories; undefined metrics N/A",
            "ap_definition": "Original tools.Evaluator trapezoidal PR area",
            "baseline_tolerances": {"metric_percentage_points": args.baseline_metric_atol,
                                    "map_atol": args.baseline_map_atol, "map_rtol": args.baseline_map_rtol},
            "source_versions": source["config"].get("versions"),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
            "reference_selection": "Exact recorded TRAIN references; no sampler invoked",
            "visualization_scale": "fixed [0,1] across panels and shots; numerical scores unchanged",
        })
        write_json(output / "diagnostic_config.json", {
            "samples": source["diagnostics"], "layers": source["config"]["features_list"],
            "layer_source": "Original PQAdapter.forward local_scores, anomaly channel 1",
            "statistics_stages": ["raw", "evaluated"], "sigma": source["config"]["sigma"],
            "warning": "Four selected samples are diagnostics, not dataset performance estimates."})
        (output / "reference_samples.txt").write_text("".join(
            f"{category}\t{path}\n" for category, path in source["reference_paths"]), encoding="utf-8")
        # Old manifests contain output paths relative to the OLD run. Preserve
        # them with an explicit name instead of claiming they are new outputs.
        write_csv(output / "source_sample_manifest.csv", list(source["manifest"][0]), source["manifest"])
        write_csv(output / "diagnostic_samples.csv", ["Category", "Sample Type", "Image Path", "Sample Key"],
                  [{key: row[key] for key in ("Category", "Sample Type", "Image Path", "Sample Key")}
                   for row in source["diagnostics"]])
        status = run_inference(args, source, output, logger)
        write_json(output / "run_status.json", status)
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error),
                                               "note": "Existing outputs are partial; do not treat as completed experiment."})
        logger.exception("Run stopped; existing output is incomplete.")
        raise
    finally:
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            sys.modules["torch"].cuda.empty_cache()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def aggregate(session, shots):
    """Collect completed workers only; include absolute deltas against each baseline."""
    fields = [f"{level}-{metric}" for level in ("I", "P") for metric in METRICS]
    rows, deltas = [], []
    for shot in shots:
        records = read_csv(session / f"visa_{shot}shot_seed10" / "metrics_combined.csv")
        baseline = {row["Category"]: row for row in records if row["Condition"] == "baseline"}
        for row in records:
            rows.append({"Shot": shot, **row})
            delta = {"Shot": shot, "Category": row["Category"], "Condition": row["Condition"]}
            for metric in fields:
                value, previous = row[metric], baseline[row["Category"]][metric]
                delta[metric] = float(value) - float(previous) if value != "N/A" and previous != "N/A" else None
            deltas.append(delta)
    columns = ["Shot", "Category", "Condition", *fields]
    write_csv(session / "metrics_summary.csv", columns, [r for r in rows if r["Category"] == "Mean"])
    write_csv(session / "metrics_by_category.csv", columns, [r for r in rows if r["Category"] != "Mean"])
    write_csv(session / "metrics_vs_baseline.csv", columns, deltas)


def main(argv=None):
    args = parse_args(argv)
    if args._worker_session:
        run_worker(args)
        return
    shots = [args.k_shots] if args.k_shots else [1, 2, 4]
    sources = [audit_source(args.source_root / f"visa_{shot}shot_seed10", shot, PROJECT_ROOT) for shot in shots]
    audit = audit_sequence(sources)
    session = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "visa_seed10")
    write_json(session / "input_audit.json", audit)
    write_json(session / "run_status.json", {"status": "running", "complete": False, "shots": shots})
    print(f"Results: {session}", flush=True)
    try:
        for shot in shots:
            command = [sys.executable, str(Path(__file__).resolve()), "--k_shots", str(shot),
                       "--source_root", str(args.source_root), "--num_workers", str(args.num_workers),
                       "--baseline_metric_atol", str(args.baseline_metric_atol),
                       "--baseline_map_atol", str(args.baseline_map_atol),
                       "--baseline_map_rtol", str(args.baseline_map_rtol),
                       "--_worker_session", str(session)]
            if args.save_maps:
                command.append("--save_maps")
            if not args.save_visualizations:
                command.append("--no-save_visualizations")
            print(f"Starting {shot}-shot; conditions: {', '.join(CONDITIONS)}", flush=True)
            with subprocess.Popen(command, cwd=str(PROJECT_ROOT)) as child:
                try:
                    code = child.wait()
                except KeyboardInterrupt:
                    try:
                        child.wait(timeout=15)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        child.terminate()
                        child.wait()
                    raise
            if code:
                raise RuntimeError(f"{shot}-shot worker exited with code {code}; remaining shots stopped.")
            aggregate(session, shots[:shots.index(shot) + 1])
        write_json(session / "run_status.json", {"status": "completed", "complete": True, "shots": shots})
    except BaseException as error:
        write_json(session / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                "complete": False, "error": str(error)})
        raise
    print(f"Completed. Summary: {session / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
