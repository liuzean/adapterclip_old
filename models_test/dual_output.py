"""VisA: baseline image score plus no_TA pixel map, with recorded-run parity.

Project root: python models_test/dual_output.py
Default: sequential isolated 1/2/4-shot workers, seed=10, no training.
"""

import argparse
import gc
import logging
import math
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from branch_ablation import infer_batch
from branch_ablation_utils import select_samples
from fusion_ablation import aggregate, initialize
from fusion_ablation_utils import (
    audit_sequence, audit_source, compare_baseline, metric_tables, new_directory,
    path_key, project_path, read_json, save_panels, write_csv, write_json,
)
from dual_output_utils import (
    BASE_CONDITIONS, CONDITIONS, FIELDS, CategoryBuffer, audit_ablation,
    compare_ablation, compare_scores, route_outputs,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/branch_diagnostics")
    parser.add_argument("--ablation_root", default="results/branch_ablation/visa_seed10",
                        help="Completed branch_ablation session used for parity, not parameter tuning.")
    parser.add_argument("--k_shots", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_per_type", type=int, default=1, help="First N normal/anomalous images per class.")
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true", help="Selected samples, two unique raw/evaluated map sets only.")
    parser.add_argument("--disable_dual_output", action="store_true", help="Route both candidate outputs to baseline.")
    parser.add_argument("--baseline_metric_atol", type=float, default=0.001, help="Percentage points.")
    parser.add_argument("--baseline_map_atol", type=float, default=1e-6)
    parser.add_argument("--baseline_map_rtol", type=float, default=1e-5)
    parser.add_argument("--score_atol", type=float, default=1e-6)
    parser.add_argument("--score_rtol", type=float, default=1e-5)
    parser.add_argument("--_worker_session", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.vis_per_type < 1 or any(not math.isfinite(value) or value < 0 for value in (
            args.baseline_metric_atol, args.baseline_map_atol, args.baseline_map_rtol, args.score_atol, args.score_rtol)):
        parser.error("Counts and finite tolerances must be nonnegative; vis_per_type must be positive.")
    if args._worker_session and args.k_shots is None:
        parser.error("Internal worker requires an explicit shot count.")
    args.source_root = project_path(args.source_root, PROJECT_ROOT)
    args.ablation_root = project_path(args.ablation_root, PROJECT_ROOT)
    return args


def save_sample(args, source, output, sample, image, mask, raw, maps, index):
    """Save aligned displays and optional unique arrays; compare original FINAL.

    Baseline scores still depend on the baseline map. A candidate localization
    map must never replace that map when reconstructing an image score.
    """
    import numpy as np
    raw_arrays = {name: raw[name][index].detach().cpu().numpy() for name in BASE_CONDITIONS}
    evaluated = {name: maps[name][index].numpy() for name in BASE_CONDITIONS}
    relative = Path(sample["Category"]) / sample["Sample Type"] / sample["Sample Key"]
    if args.save_visualizations:
        save_panels(output / "visualizations" / relative.with_suffix(".png"), image, mask.numpy(),
                    raw_arrays, evaluated, list(BASE_CONDITIONS), 2,
                    f"VisA {source['config']['k_shots']}-shot / {sample['Category']} / {sample['Sample Key']} / "
                    f"dual pixel={'baseline' if args.disable_dual_output else 'no_TA'}; image=baseline")
    checks = []
    for stage, field, arrays in (("raw", "Raw Maps", raw_arrays), ("evaluated", "Evaluated Maps", evaluated)):
        if args.save_maps:
            directory = output / "raw_maps" / relative / stage
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in arrays.items():
                np.save(directory / f"{name}.npy", value, allow_pickle=False)
        old = Path(source["source"]) / sample[field] / "FINAL.npy" if sample.get(field) else None
        check = {"Image Path": sample["Image Path"], "Stage": stage, "Reference": str(old) if old else "",
                 "Status": "unavailable", "Max Abs Delta": None}
        if old is not None and old.is_file():
            previous = np.load(old, allow_pickle=False)
            value = arrays["baseline"]
            same_shape = previous.shape == value.shape
            passed = same_shape and np.isfinite(previous).all() and np.allclose(
                previous, value, atol=args.baseline_map_atol, rtol=args.baseline_map_rtol)
            check.update({"Status": "passed" if passed else "failed",
                          "Max Abs Delta": float(np.max(np.abs(previous - value))) if same_shape else None})
        checks.append(check)
    return checks


def run_inference(args, source, recorded, output, logger):
    """One original adapter pass per batch; route its outputs before evaluation."""
    import torch
    from tools import setup_seed
    from test import prompt_association
    from tqdm import tqdm
    setup_seed(source["config"]["seed"])
    config, model, textual, visual, pq, learned_text, memory, loader, dpam, device = initialize(
        source, output, args.num_workers, logger)
    selected = {path_key(row["Image Path"]): row for row in source["diagnostics"]}
    pixels, images, score_rows, score_checks, map_checks, routing_checks = [], [], [], [], [], []
    buffer, processed, closed, visited = None, 0, set(), set()

    def save_checks():
        for filename, rows in (("image_scores.csv", score_rows), ("image_score_parity.csv", score_checks),
                               ("baseline_map_checks.csv", map_checks), ("routing_checks.csv", routing_checks)):
            if rows:
                write_csv(output / filename, list(rows[0]), rows)

    def check_metrics(combined):
        baseline = compare_baseline(source, combined, args.baseline_metric_atol)
        ablation = compare_ablation(recorded, combined, args.baseline_metric_atol)
        write_csv(output / "baseline_reference_comparison.csv", list(baseline[0]), baseline)
        write_csv(output / "ablation_reference_comparison.csv", list(ablation[0]), ablation)
        if not all(row["Passed"] for row in [*baseline, *ablation]):
            raise RuntimeError("Recorded baseline/no_TA metric parity failed; remaining shots stopped.")

    def flush(current):
        p_rows, i_rows = current.finish(device, logger)
        pixels.extend(p_rows)
        images.extend(i_rows)
        combined = metric_tables(output, pixels, images)
        save_checks()
        # Partial-class means cannot be compared with full-dataset macro means.
        check_metrics([row for row in combined if row["Category"] != "Mean"])
        gc.collect()
        torch.cuda.empty_cache()

    for items in tqdm(loader, desc=f"Dual output {config.k_shots}-shot"):
        paths = items["img_path"]
        with torch.no_grad():
            features, patches = model.encode_image(items["img"].to(device), config.features_list, DPAM_layer=dpam)
            prompt_features, prompt_patches = prompt_association(*memory, items["cls_name"])
            raw, evaluated, scores, maxima, globals_, fixed_global, extras, distances = infer_batch(
                config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths, nullcontext())
        # Reuse the validated branch-ablation extraction; discard its no_VA and
        # layer diagnostics immediately, without hooks or another forward.
        del extras, distances
        for values in (raw, evaluated, scores, maxima):
            values.pop("no_VA")
        maps, routed_scores = route_outputs(evaluated, scores, args.disable_dual_output)
        pixel_source = "baseline" if args.disable_dual_output else "no_TA"
        routing_checks.append({"Batch": len(routing_checks), "Sample Count": len(paths),
            "Pixel Source": pixel_source, "Image Source": "baseline",
            "Pixel Equal": torch.equal(maps["dual_output"], evaluated[pixel_source]),
            "Image Equal": torch.equal(routed_scores["dual_output"], scores["baseline"])})
        if not routing_checks[-1]["Pixel Equal"] or not routing_checks[-1]["Image Equal"]:
            save_checks()
            raise RuntimeError("Dual-output routing changed a source output.")
        masks = items["img_mask"][:, 0].clone()
        masks[masks > 0.5], masks[masks <= 0.5] = 1, 0
        for name in CONDITIONS:
            if maps[name].shape != masks.shape or routed_scores[name].shape != (len(paths),):
                raise ValueError(f"Unexpected {name} map/score shape.")
        for index, category in enumerate(items["cls_name"]):
            path, label = paths[index], int(items["anomaly"][index])
            if processed >= len(source["manifest"]):
                raise ValueError("Extra runtime test images.")
            expected = source["manifest"][processed]
            if (path_key(path) != path_key(expected["Image Path"]) or category != expected["Category"]
                    or label != int(expected["Sample Type"] == "Anomaly")):
                raise ValueError(f"Runtime test identity/order/label differs: {path}")
            if buffer is None or buffer.category != category:
                if buffer is not None:
                    flush(buffer)
                    closed.add(buffer.category)
                if category in closed:
                    raise ValueError("Expected contiguous test classes for bounded buffering.")
                buffer = CategoryBuffer(category, args.disable_dual_output)
            buffer.append(masks[index], label, maps, routed_scores, index)
            row = {"Category": category, "Image Path": path, "Label": label,
                   **{f"{name} Global": float(value[index]) for name, value in globals_.items()},
                   "Fixed Global Mean": float(fixed_global[index])}
            for name in BASE_CONDITIONS:
                row.update({f"{name} Map Max": float(maxima[name][index]), f"{name} Image Score": float(scores[name][index])})
            row.update({"dual_output Pixel Map Max": float(maxima[pixel_source][index]),
                        "dual_output Image Score": float(routed_scores["dual_output"][index]),
                        "dual_output Pixel Source": pixel_source, "dual_output Image Source": "baseline"})
            score_rows.append(row)
            score_checks.append(compare_scores(row, recorded["scores"][processed], args.score_atol, args.score_rtol))
            if not score_checks[-1]["Passed"]:
                save_checks()
                raise RuntimeError(f"Recorded image score parity failed: {path}")
            key = path_key(path)
            if key in selected:
                checks = save_sample(args, source, output, selected[key], items["img"][index], masks[index], raw, maps, index)
                map_checks.extend(checks)
                visited.add(key)
                if any(check["Status"] == "failed" for check in checks):
                    save_checks()
                    raise RuntimeError("Original FINAL map parity failed.")
            processed += 1
        del features, patches, prompt_features, prompt_patches, raw, evaluated, scores, maxima, globals_, fixed_global, maps, routed_scores
    if buffer is None or processed != len(source["manifest"]) or visited != set(selected):
        raise ValueError("Incomplete test or visualization coverage.")
    flush(buffer)
    combined = metric_tables(output, pixels, images)
    check_metrics(combined)
    write_csv(output / "metrics_combined.csv", ["Category", "Condition", *FIELDS], combined)
    return {"status": "completed", "complete": True, "sample_count": processed, "category_count": len(closed) + 1,
            "conditions": list(CONDITIONS), "visualization_sample_count": len(visited),
            "pixel_source": pixel_source, "image_source": "baseline", "routing": "exact_outputs_shared",
            "baseline_metrics": "passed_within_tolerance", "ablation_metrics": "passed_within_tolerance",
            "image_score_parity": {"passed": len(score_checks), "failed": 0},
            "baseline_map_checks": {state: sum(row["Status"] == state for row in map_checks)
                                    for state in ("passed", "failed", "unavailable")},
            "fresh_test_py_rerun": False, "pqa_internal_hooks": False}


def run_worker(args):
    session = Path(args._worker_session).resolve()
    session.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    output = session / f"visa_{args.k_shots}shot_seed10"
    output.mkdir()
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    logger = logging.getLogger("dual_output")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        source = audit_source(args.source_root / f"visa_{args.k_shots}shot_seed10", args.k_shots, PROJECT_ROOT)
        selection = select_samples([source], args.vis_per_type)
        recorded = audit_ablation(source, args.ablation_root / f"visa_{args.k_shots}shot_seed10")
        audit = read_json(session / "input_audit.json")
        snapshot = next(row for row in audit["runs"] if row["shot"] == args.k_shots)
        previous = next(row for row in audit["ablation_runs"] if row["shot"] == args.k_shots)
        if (source["hashes"] != snapshot["source_file_sha256"] or recorded["hashes"] != previous["file_sha256"]
                or selection["samples"] != audit["visualization_selection"]["samples"]
                or args.disable_dual_output != audit["disable_dual_output"]):
            raise ValueError("Inputs/routing changed after parent audit.")
        import torch
        import numpy as np
        import scipy
        write_json(output / "experiment_config.json", {
            **source["config"], "source_result_dir": source["source"], "ablation_result_dir": recorded["directory"],
            "result_dir": str(output), "project_root": str(PROJECT_ROOT), "num_workers": args.num_workers,
            "save_maps": args.save_maps, "save_visualizations": args.save_visualizations,
            "vis_per_class": args.vis_per_type, "visualization_sample_count": len(source["diagnostics"]),
            "disable_dual_output": args.disable_dual_output, "conditions": list(CONDITIONS),
            "pixel_source": "baseline" if args.disable_dual_output else "no_TA", "image_source": "baseline",
            "native_global_policy": "Original mean(VA_global,TA_global,PQA_global); all three adapters retained",
            "image_score_policy": "Original harmonic(global_mean, max(smoothed_baseline_map)); never use dual pixel maximum",
            "pixel_policy": "Original smoothed H(mean(VA,PQA),ALIGN); baseline when disabled",
            "maps_scope": "Selected samples, baseline/no_TA raw and evaluated arrays; dual output aliases pixel_source",
            "statistics_stage": "Per-image score components only; no PQA layer/internal statistics",
            "visualization": "Fixed [0,1]; two unique maps, input and GT; no score normalization",
            "baseline_verification": "Original metrics/selected FINAL maps; previous baseline/no_TA metrics and per-image components",
            "source_versions": source["config"].get("versions"),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
            "tolerances": {"metric_pp": args.baseline_metric_atol, "map_atol": args.baseline_map_atol,
                           "map_rtol": args.baseline_map_rtol, "score_atol": args.score_atol, "score_rtol": args.score_rtol},
            "reference_selection": "Exact recorded normal TRAIN references, original order and encoding batches",
            "evaluation_scope": "Full VisA; exploratory routing verification, not independent generalization evidence",
        })
        (output / "reference_samples.txt").write_text("".join(f"{c}\t{p}\n" for c, p in source["reference_paths"]), encoding="utf-8")
        write_csv(output / "source_sample_manifest.csv", list(source["manifest"][0]), source["manifest"])
        write_csv(output / "visualization_samples.csv", list(selection["samples"][0]), selection["samples"])
        write_json(output / "run_status.json", run_inference(args, source, recorded, output, logger))
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error), "note": "Existing outputs may be incomplete."})
        logger.exception("Run stopped.")
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
    audit["visualization_selection"] = select_samples(sources, args.vis_per_type)
    audit["disable_dual_output"] = args.disable_dual_output
    audit["ablation_runs"] = []
    for source, shot in zip(sources, shots):
        recorded = audit_ablation(source, args.ablation_root / f"visa_{shot}shot_seed10")
        audit["ablation_runs"].append({"shot": shot, "directory": recorded["directory"], "file_sha256": recorded["hashes"]})
    session = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "visa_seed10")
    write_json(session / "input_audit.json", audit)
    write_json(session / "run_status.json", {"status": "running", "complete": False, "shots": shots})
    print(f"Results: {session}", flush=True)
    try:
        for position, shot in enumerate(shots):
            command = [sys.executable, str(Path(__file__).resolve()), "--k_shots", str(shot),
                       "--source_root", str(args.source_root), "--ablation_root", str(args.ablation_root),
                       "--num_workers", str(args.num_workers), "--vis_per_type", str(args.vis_per_type),
                       "--_worker_session", str(session)]
            for name in ("baseline_metric_atol", "baseline_map_atol", "baseline_map_rtol", "score_atol", "score_rtol"):
                command.extend([f"--{name}", str(getattr(args, name))])
            for name in ("save_maps", "disable_dual_output"):
                if getattr(args, name):
                    command.append(f"--{name}")
            if not args.save_visualizations:
                command.append("--no-save_visualizations")
            print(f"Starting {shot}-shot: {', '.join(CONDITIONS)}", flush=True)
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
            status = read_json(session / f"visa_{shot}shot_seed10" / "run_status.json")
            if status.get("status") != "completed" or status.get("complete") is not True:
                raise RuntimeError(f"{shot}-shot did not complete; remaining shots stopped.")
            aggregate(session, shots[:position + 1])
        write_json(session / "run_status.json", {"status": "completed", "complete": True, "shots": shots})
    except BaseException as error:
        write_json(session / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error)})
        raise
    print(f"Completed. Summary: {session / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
