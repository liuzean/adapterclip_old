"""VisA single-branch pixel ablations and 12-class PQA diagnostics.

From project root: python models_test/branch_ablation.py
Default: isolated sequential 1/2/4-shot workers, seed=10, no training.
"""

import argparse
import gc
import logging
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_ablation import aggregate, initialize
from fusion_ablation_utils import (
    METRICS, audit_sequence, audit_source, compare_baseline, metric_tables,
    new_directory, path_key, project_path, read_json, save_panels, write_csv, write_json,
)
from branch_ablation_utils import (
    CONDITIONS, INTERNAL_FIELDS, LAYER_STAT_FIELDS, CategoryBuffer, PQALayerObserver,
    compute_conditions, layer_summary, map_statistics, select_samples,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/branch_diagnostics")
    parser.add_argument("--k_shots", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--diagnostic_per_type", type=int, default=3,
                        help="First N normal and N anomalous source-manifest images per class; default 72 images/shot.")
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true", help="Save maps for selected diagnostic samples only.")
    parser.add_argument("--baseline_metric_atol", type=float, default=0.001, help="Metric tolerance in percentage points.")
    parser.add_argument("--baseline_map_atol", type=float, default=1e-6)
    parser.add_argument("--baseline_map_rtol", type=float, default=1e-5)
    parser.add_argument("--_worker_session", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.diagnostic_per_type < 1 or any(
            not math.isfinite(value) or value < 0 for value in (
                args.baseline_metric_atol, args.baseline_map_atol, args.baseline_map_rtol)):
        parser.error("Counts and finite tolerances must be nonnegative; diagnostic_per_type must be positive.")
    if args._worker_session and args.k_shots is None:
        parser.error("Internal worker requires an explicit shot count.")
    args.source_root = project_path(args.source_root, PROJECT_ROOT)
    return args


def infer_batch(config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths, observer):
    """Mirror stable score extraction, with three controlled local fusions.

    Globals [B,D], four patches [B,1+L,D], prompt patches [B,S,1+L,D].
    Original PQAdapter returns four [B,2,H,W] probabilities and four distances
    [B,1,H,W]. Maps returned here are [B,H,W]; image scores are [B].
    Only score extraction/aggregation mirrors test.py/fusion_ablation.infer_batch;
    actual Adapter forwards, Top-1 matching and residuals remain original code.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import require_finite, smooth_maps
    va_global, va_local = visual(features, patches, textual.static_text_features)
    ta_global, ta_local = textual.compute_global_local_score(features, patches, learned_text)
    with observer:
        pq_global, pq_local, align_local = pq(features, patches, prompt_features, prompt_patches)
    if len(pq_local) != 4 or len(align_local) != 4:
        raise ValueError("Expected original four PQA probability/distance maps.")
    layers = {f"PQA_L{layer}": value[:, 1].detach()
              for layer, value in zip(config.features_list, pq_local)}
    pqa = torch.concat([value.unsqueeze(1) for value in layers.values()], dim=1).mean(dim=1).detach()
    align = fusion_fun(align_local, fusion_type="harmonic_mean")[:, 0]
    distances = {f"ALIGN_L{layer}": value[:, 0].detach()
                 for layer, value in zip(config.features_list, align_local)}
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
    for name, value in {**layers, **distances, **globals_, "PQA_Fused": pqa,
                        "ALIGN": align, "Fixed Global Mean": fixed_global}.items():
        require_finite(name, value, paths)
    for name in CONDITIONS:
        require_finite(f"{name} raw", raw[name], paths)
        smoothed = smooth_maps(raw[name], config.sigma).to(features.device)
        maximum = torch.max(smoothed.view(features.shape[0], -1), dim=1)[0]
        score = fusion_fun([fixed_global, maximum], fusion_type="harmonic_mean")
        require_finite(f"{name} smoothed", smoothed, paths)
        require_finite(f"{name} image score", score, paths)
        evaluated[name] = torch.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0).cpu()
        scores[name] = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).cpu()
        maxima[name] = maximum.cpu()
    return raw, evaluated, scores, maxima, globals_, fixed_global, {**layers, "PQA_Fused": pqa, "ALIGN": align}, distances


def diagnose_sample(args, source, output, sample, image, mask, raw, evaluated, extras, distances, index):
    """Inspect one fixed sample; never use these subset rows as dataset metrics.

    Reuse original statistics, smoothing and plotting helpers. Save raw/evaluated
    PQA layers, fused PQA/ALIGN and conditions optionally. Per-layer distances
    are summarized without duplicating their full matrices on disk.
    """
    import numpy as np
    import torch
    from branch_diagnostics_utils import require_finite, smooth_maps
    maps = {name: value[index].detach().cpu() for name, value in {**extras, **raw, **distances}.items()}
    for name, value in maps.items():
        if value.shape != mask.shape:
            raise ValueError(f"{name} shape {value.shape} != GT {mask.shape}")
        require_finite(name, value, [sample["Image Path"]])
    reconstructed = torch.concat([maps[f"PQA_L{layer}"].unsqueeze(0)
                                  for layer in source["config"]["features_list"]], dim=0).mean(dim=0)
    if not torch.allclose(reconstructed, maps["PQA_Fused"], atol=1e-6, rtol=1e-5):
        raise ValueError("Extracted four layers do not reconstruct original PQA within tolerance.")
    raw_arrays = {name: value.numpy() for name, value in maps.items()}
    smooth_arrays = {name: (evaluated[name][index] if name in evaluated else
                            smooth_maps(value.unsqueeze(0), source["config"]["sigma"])[0]).numpy()
                     for name, value in maps.items()}
    mask_array = mask.numpy()
    rows = [map_statistics(source["config"]["k_shots"], sample, name, stage, value, mask_array)
            for stage, arrays in (("raw", raw_arrays), ("evaluated", smooth_arrays))
            for name, value in arrays.items()]
    relative = Path(sample["Category"]) / sample["Sample Type"] / sample["Sample Key"]
    if args.save_visualizations:
        title = f"VisA {source['config']['k_shots']}-shot seed=10 / {sample['Category']} / {sample['Sample Key']}"
        save_panels(output / "visualizations/pqa_layers" / relative.with_suffix(".png"),
                    image, mask_array, raw_arrays, smooth_arrays,
                    [f"PQA_L{layer}" for layer in source["config"]["features_list"]]
                    + ["PQA_Fused", "ALIGN", "baseline"], 3, title)
        # Two rows avoid the title collision seen in the old horizontal plot.
        save_panels(output / "visualizations/branch_conditions" / relative.with_suffix(".png"),
                    image, mask_array, raw_arrays, smooth_arrays, list(CONDITIONS), 3, title)
    if args.save_maps:
        for stage, arrays in (("raw", raw_arrays), ("evaluated", smooth_arrays)):
            directory = output / "raw_maps" / relative / stage
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in arrays.items():
                if not name.startswith("ALIGN_L"):
                    np.save(directory / f"{name}.npy", value, allow_pickle=False)
    checks = []
    for stage, field, arrays in (("raw", "Raw Maps", raw_arrays), ("evaluated", "Evaluated Maps", smooth_arrays)):
        for current, original in (("PQA_Fused", "PQA"), ("ALIGN", "ALIGN"), ("baseline", "FINAL")):
            old_directory = sample.get(field)
            path = Path(source["source"]) / old_directory / f"{original}.npy" if old_directory else None
            row = {"Image Path": sample["Image Path"], "Stage": stage, "Map": original,
                   "Reference": str(path) if path else "", "Status": "unavailable", "Max Abs Delta": None}
            if path is not None and path.is_file():
                previous = np.load(path, allow_pickle=False)
                value = arrays[current]
                compatible = previous.shape == value.shape
                passed = compatible and np.isfinite(previous).all() and np.allclose(
                    previous, value, atol=args.baseline_map_atol, rtol=args.baseline_map_rtol)
                row.update({"Status": "passed" if passed else "failed",
                            "Max Abs Delta": float(np.max(np.abs(previous - value))) if compatible else None})
            checks.append(row)
    return rows, checks


def save_diagnostics(output, statistics, internals, map_checks):
    """Write small scalar tables; raw layers are never accumulated per class."""
    groups = {
        "pqa_layer_statistics.csv": [row for row in statistics if row["Map"].startswith("PQA_L")],
        "alignment_layer_statistics.csv": [row for row in statistics if row["Map"].startswith("ALIGN_L")],
        "diagnostic_statistics.csv": [row for row in statistics if not row["Map"].startswith(("PQA_L", "ALIGN_L"))],
    }
    for filename, rows in groups.items():
        write_csv(output / filename, LAYER_STAT_FIELDS, rows)
    write_csv(output / "pqa_internal_statistics.csv", INTERNAL_FIELDS, internals)
    summary = layer_summary(groups["pqa_layer_statistics.csv"])
    if summary:
        write_csv(output / "pqa_layer_summary.csv", list(summary[0]), summary)
    if map_checks:
        write_csv(output / "baseline_map_checks.csv", list(map_checks[0]), map_checks)


def run_inference(args, source, output, logger):
    """One full pass per shot serves both ablations and fixed 12-class diagnostics."""
    import torch
    from tools import setup_seed
    from test import prompt_association
    from tqdm import tqdm
    setup_seed(source["config"]["seed"])
    config, model, textual, visual, pq, learned_text, memory, loader, dpam, device = initialize(
        source, output, args.num_workers, logger)
    selected = {path_key(row["Image Path"]): row for row in source["diagnostics"]}
    pixels, images, statistics, internals, map_checks, score_rows, metric_checks = [], [], [], [], [], [], []
    buffer, processed, closed, visited = None, 0, set(), set()

    def flush(current):
        p_rows, i_rows = current.finish(device, logger)
        pixels.extend(p_rows)
        images.extend(i_rows)
        combined = metric_tables(output, pixels, images)
        checks = compare_baseline(source, [row for row in combined if row["Category"] == current.category], args.baseline_metric_atol)
        metric_checks.extend(checks)
        write_csv(output / "baseline_reference_comparison.csv", list(metric_checks[0]), metric_checks)
        write_csv(output / "image_scores.csv", list(score_rows[0]), score_rows)
        save_diagnostics(output, statistics, internals, map_checks)
        if not all(row["Passed"] for row in checks):
            raise RuntimeError(f"Baseline metrics differ for {current.category}; remaining shots stopped.")
        gc.collect()
        torch.cuda.empty_cache()

    for items in tqdm(loader, desc=f"Branch ablation {config.k_shots}-shot"):
        paths = items["img_path"]
        selection = {index: selected[path_key(path)] for index, path in enumerate(paths) if path_key(path) in selected}
        observer = PQALayerObserver(pq, config.features_list, config.k_shots, selection)
        with torch.no_grad():
            features, patches = model.encode_image(items["img"].to(device), config.features_list, DPAM_layer=dpam)
            prompt_features, prompt_patches = prompt_association(*memory, items["cls_name"])
            raw, evaluated, scores, maxima, globals_, fixed_global, extras, distances = infer_batch(
                config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths, observer)
        masks = items["img_mask"][:, 0].clone()
        masks[masks > 0.5], masks[masks <= 0.5] = 1, 0
        for name in CONDITIONS:
            if evaluated[name].shape != masks.shape or scores[name].shape != (len(paths),):
                raise ValueError(f"Unexpected {name} map/score shape for this batch.")
        for index, category in enumerate(items["cls_name"]):
            path = paths[index]
            label = int(items["anomaly"][index])
            if processed >= len(source["manifest"]):
                raise ValueError("Runtime test set contains extra images.")
            expected = source["manifest"][processed]
            if (path_key(path) != path_key(expected["Image Path"]) or category != expected["Category"]
                    or label != int(expected["Sample Type"] == "Anomaly")):
                raise ValueError(f"Test identity/order/label differs from source: {path}")
            if buffer is None or buffer.category != category:
                if buffer is not None:
                    flush(buffer)
                    closed.add(buffer.category)
                if category in closed:
                    raise ValueError("Test classes must be contiguous for bounded category buffering.")
                buffer = CategoryBuffer(category)
            buffer.append(masks[index], label, evaluated, scores, index)
            score_row = {"Category": category, "Image Path": path, "Label": label,
                         **{f"{name} Global": float(value[index]) for name, value in globals_.items()},
                         "Fixed Global Mean": float(fixed_global[index])}
            for name in CONDITIONS:
                score_row.update({f"{name} Map Max": float(maxima[name][index]), f"{name} Image Score": float(scores[name][index])})
            score_rows.append(score_row)
            if index in selection:
                rows, checks = diagnose_sample(args, source, output, selection[index], items["img"][index],
                                               masks[index], raw, evaluated, extras, distances, index)
                statistics.extend(rows)
                captured = [row for row in observer.rows if path_key(row["Image Path"]) == path_key(path)]
                if len(captured) != 4 * 4:
                    raise ValueError(f"Expected four observation stages per PQA layer: {path}")
                internals.extend(captured)
                map_checks.extend(checks)
                visited.add(path_key(path))
                if any(row["Status"] == "failed" for row in checks):
                    save_diagnostics(output, statistics, internals, map_checks)
                    raise RuntimeError("Stored PQA/ALIGN/baseline map mismatch; inspect baseline_map_checks.csv.")
            processed += 1
        del observer, selection, features, patches, prompt_features, prompt_patches, raw, evaluated, scores, maxima, globals_, fixed_global, extras, distances
    if buffer is None or processed != len(source["manifest"]) or visited != set(selected):
        raise ValueError("Incomplete test or fixed diagnostic coverage.")
    if len(internals) != len(selected) * 4 * 4:
        raise ValueError("Incomplete PQA hook observations.")
    flush(buffer)
    combined = metric_tables(output, pixels, images)
    checks = compare_baseline(source, combined, args.baseline_metric_atol)
    write_csv(output / "baseline_reference_comparison.csv", list(checks[0]), checks)
    if not all(row["Passed"] for row in checks):
        raise RuntimeError("Baseline macro/category metrics differ; remaining shots stopped.")
    fields = [f"{level}-{metric}" for level in ("I", "P") for metric in METRICS]
    write_csv(output / "metrics_combined.csv", ["Category", "Condition", *fields], combined)
    counts = Counter((row["Category"], row["Sample Type"]) for row in source["diagnostics"])
    write_json(output / "diagnostic_coverage.json", {"complete": True, "expected": len(selected), "observed": len(visited),
        "internal_rows": len(internals), "per_class_type": [{"Category": c, "Sample Type": label, "Count": n}
                                                            for (c, label), n in counts.items()]})
    return {"status": "completed", "complete": True, "sample_count": processed, "category_count": len(closed) + 1,
            "conditions": list(CONDITIONS), "diagnostic_sample_count": len(visited),
            "baseline_metrics": "passed_within_tolerance", "pqa_layer_reconstruction": "passed_within_tolerance",
            "pqa_internal_hooks": "all_expected_real_layer_calls_observed_and_hooks_removed",
            "baseline_map_checks": {state: sum(row["Status"] == state for row in map_checks)
                                    for state in ("passed", "failed", "unavailable")}, "fresh_test_py_rerun": False}


def run_worker(args):
    session = Path(args._worker_session).resolve()
    session.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    output = session / f"visa_{args.k_shots}shot_seed10"
    output.mkdir()
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    logger = logging.getLogger("branch_ablation")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        source = audit_source(args.source_root / f"visa_{args.k_shots}shot_seed10", args.k_shots, PROJECT_ROOT)
        selection = select_samples([source], args.diagnostic_per_type)
        parent_audit = read_json(session / "input_audit.json")
        snapshot = next(row for row in parent_audit["runs"] if row["shot"] == args.k_shots)
        if source["hashes"] != snapshot["source_file_sha256"] or selection["samples"] != parent_audit["diagnostic_selection"]["samples"]:
            raise ValueError("Source inputs changed after parent audit; refusing an inconsistent run.")
        import torch
        import numpy as np
        import scipy
        write_json(output / "experiment_config.json", {
            **source["config"], "source_result_dir": source["source"], "result_dir": str(output),
            "num_workers": args.num_workers, "save_visualizations": args.save_visualizations,
            "save_maps": args.save_maps, "maps_scope": "selected_diagnostics_only",
            "diagnostic_per_class_per_type": args.diagnostic_per_type, "diagnostic_sample_count": len(source["diagnostics"]),
            "vis_per_class": args.diagnostic_per_type, "project_root": str(PROJECT_ROOT),
            "conditions": list(CONDITIONS), "native_global_policy": "Fixed original mean(VA,TA,PQA) for every condition",
            "source_versions": source["config"].get("versions"),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
            "baseline_tolerances": {"metric_percentage_points": args.baseline_metric_atol,
                                    "map_atol": args.baseline_map_atol, "map_rtol": args.baseline_map_rtol},
            "reference_selection": "Exact recorded normal TRAIN references, original order and encoding batches",
            "visualization_scale": "Fixed [0,1], no score normalization or clipping",
            "visualization": "Fixed [0,1] across samples/shots; source display policy is not reused",
            "statistics_stage": "raw and evaluated, per selected image; category summary over fixed selected images",
            "baseline_verification": "Per-category and macro full-precision CSV comparison; selected raw/evaluated original map checks",
            "diagnostic_statistics_scope": "Fixed selected images only; not full-dataset layer metrics",
        })
        write_json(output / "diagnostic_config.json", {**selection, "features_list": source["config"]["features_list"],
            "internal_stages": ["pq_context_before_bn", "pq_context_after_bn", "local_logit_difference", "local_probability_before_resize"],
            "pq_context_input": "Original query + abs(query-aligned_prompt); not an isolated residual measurement",
            "alignment_layer_maps": "Original PQAdapter returned interpolated distances; no matching recomputation",
            "source": "Original PQAdapter outputs and temporary non-mutating BN/unique-head forward hooks"})
        (output / "reference_samples.txt").write_text("".join(f"{category}\t{path}\n" for category, path in source["reference_paths"]), encoding="utf-8")
        write_csv(output / "diagnostic_samples.csv", ["Category", "Sample Type", "Image Path", "Sample Key"], selection["samples"])
        write_csv(output / "source_sample_manifest.csv", list(source["manifest"][0]), source["manifest"])
        status = run_inference(args, source, output, logger)
        write_json(output / "run_status.json", status)
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error), "note": "Any existing output is incomplete."})
        logger.exception("Run stopped; output is incomplete.")
        raise
    finally:
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            sys.modules["torch"].cuda.empty_cache()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main(argv=None):
    args = parse_args(argv)
    if args._worker_session:
        run_worker(args)
        return
    shots = [args.k_shots] if args.k_shots else [1, 2, 4]
    sources = [audit_source(args.source_root / f"visa_{shot}shot_seed10", shot, PROJECT_ROOT) for shot in shots]
    audit = audit_sequence(sources)
    audit["diagnostic_selection"] = select_samples(sources, args.diagnostic_per_type)
    session = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "visa_seed10")
    write_json(session / "input_audit.json", audit)
    write_json(session / "run_status.json", {"status": "running", "complete": False, "shots": shots})
    print(f"Results: {session}", flush=True)
    try:
        for position, shot in enumerate(shots):
            command = [sys.executable, str(Path(__file__).resolve()), "--k_shots", str(shot),
                       "--source_root", str(args.source_root), "--num_workers", str(args.num_workers),
                       "--diagnostic_per_type", str(args.diagnostic_per_type),
                       "--baseline_metric_atol", str(args.baseline_metric_atol), "--baseline_map_atol", str(args.baseline_map_atol),
                       "--baseline_map_rtol", str(args.baseline_map_rtol), "--_worker_session", str(session)]
            if args.save_maps:
                command.append("--save_maps")
            if not args.save_visualizations:
                command.append("--no-save_visualizations")
            print(f"Starting {shot}-shot: {', '.join(CONDITIONS)}; fixed diagnostics={len(sources[position]['diagnostics'])}", flush=True)
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
                raise RuntimeError(f"{shot}-shot failed with exit code {code}; remaining shots stopped.")
            aggregate(session, shots[:position + 1])
        write_json(session / "run_status.json", {"status": "completed", "complete": True, "shots": shots})
    except BaseException as error:
        write_json(session / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                "complete": False, "error": str(error)})
        raise
    print(f"Completed. Summary: {session / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
