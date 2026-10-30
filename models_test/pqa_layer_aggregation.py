"""PQA spatial-agreement aggregation: four controlled VisA conditions.

Project root: python models_test/pqa_layer_aggregation.py
Default: isolated sequential 1/2/4-shot, seed=10, one model pass per batch.
"""

import argparse
import gc
import logging
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from branch_ablation_utils import select_samples
from dual_output import save_sample as check_original_sample
from dual_output_utils import FIELDS, audit_ablation, compare_scores
from fusion_ablation import aggregate, initialize
from fusion_ablation_utils import (
    audit_sequence, audit_source, compare_baseline, metric_tables, new_directory,
    path_key, project_path, read_json, save_panels, write_csv, write_json,
)
from pqa_layer_aggregation_utils import (
    CONDITIONS, GRID_SIZE, IMAGE_SOURCES, PIXEL_SOURCES, CategoryBuffer,
    compare_controls, infer_batch,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/branch_diagnostics")
    parser.add_argument("--control_root", default="results/dual_output/visa_seed10")
    parser.add_argument("--k_shots", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument("--aggregation_strength", type=float, default=.5,
                        help="Fixed uniform/adaptive blend in [0,1]; no automatic search.")
    parser.add_argument("--disable_aggregation", action="store_true", help="Force exact original layer mean.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_per_type", type=int, default=1)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--baseline_metric_atol", type=float, default=.001)
    parser.add_argument("--baseline_map_atol", type=float, default=1e-6)
    parser.add_argument("--baseline_map_rtol", type=float, default=1e-5)
    parser.add_argument("--score_atol", type=float, default=1e-6)
    parser.add_argument("--score_rtol", type=float, default=1e-5)
    parser.add_argument("--_worker_session", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.aggregation_strength) or not 0 <= args.aggregation_strength <= 1:
        parser.error("aggregation_strength must be finite and in [0,1].")
    if args.num_workers < 0 or args.vis_per_type < 1 or any(not math.isfinite(value) or value < 0 for value in (
            args.baseline_metric_atol, args.baseline_map_atol, args.baseline_map_rtol, args.score_atol, args.score_rtol)):
        parser.error("Counts and finite tolerances must be nonnegative; vis_per_type must be positive.")
    if args._worker_session and args.k_shots is None:
        parser.error("Internal worker requires an explicit shot count.")
    args.source_root = project_path(args.source_root, PROJECT_ROOT)
    args.control_root = project_path(args.control_root, PROJECT_ROOT)
    args.effective_strength = 0. if args.disable_aggregation else args.aggregation_strength
    return args


def save_sample(args, source, output, sample, image, mask, raw, maps, index):
    """Reuse original FINAL checks; plot four unique maps and save only selections."""
    import numpy as np
    check_args = SimpleNamespace(**{**vars(args), "disable_dual_output": False, "save_visualizations": False})
    checks = check_original_sample(check_args, source, output, sample, image, mask, raw, maps, index)
    relative = Path(sample["Category"]) / sample["Sample Type"] / sample["Sample Key"]
    raw_arrays = {condition: raw[name][index].detach().cpu().numpy() for condition, name in PIXEL_SOURCES.items()}
    evaluated = {condition: maps[name][index].numpy() for condition, name in PIXEL_SOURCES.items()}
    if args.save_visualizations:
        save_panels(output / "visualizations" / relative.with_suffix(".png"), image, mask.numpy(), raw_arrays,
                    evaluated, list(CONDITIONS), 3,
                    f"{source['config']['dataset']} {source['config']['k_shots']}-shot / {sample['Category']} / {sample['Sample Key']}")
    if args.save_maps:
        for stage, arrays in (("raw", raw_arrays), ("evaluated", evaluated)):
            directory = output / "raw_maps" / relative / stage
            for condition in ("pqa_weighted", "pqa_weighted_dual"):
                np.save(directory / f"{condition}.npy", arrays[condition], allow_pickle=False)
    return checks


def run_inference(args, source, recorded, output, logger):
    """Actual inference for all four conditions; historical data is only a control."""
    import torch
    from tools import setup_seed
    from test import prompt_association
    from tqdm import tqdm
    setup_seed(source["config"]["seed"])
    config, model, textual, visual, pq, learned_text, memory, loader, dpam, device = initialize(
        source, output, args.num_workers, logger)
    selected = {path_key(row["Image Path"]): row for row in source["diagnostics"]}
    pixels, images, score_rows, score_checks, weights, map_checks, routing = [], [], [], [], [], [], []
    current, processed, closed, visited = None, 0, set(), set()

    def save_checks():
        for name, rows in (("image_scores.csv", score_rows), ("image_score_parity.csv", score_checks),
                           ("layer_weights.csv", weights), ("baseline_map_checks.csv", map_checks), ("routing_checks.csv", routing)):
            if rows:
                write_csv(output / name, list(rows[0]), rows)

    def verify_metrics(rows):
        for name, checks in (("baseline_reference_comparison.csv", compare_baseline(source, rows, args.baseline_metric_atol)),
                             ("control_reference_comparison.csv", compare_controls(recorded, rows, args.baseline_metric_atol))):
            write_csv(output / name, list(checks[0]), checks)
            if not all(row["Passed"] for row in checks):
                raise RuntimeError(f"Control parity failed: {name}; remaining shots stopped.")

    def flush(buffer):
        p_rows, i_rows = buffer.finish(device, logger)
        pixels.extend(p_rows)
        images.extend(i_rows)
        combined = metric_tables(output, pixels, images)
        save_checks()
        verify_metrics([row for row in combined if row["Category"] != "Mean"])
        gc.collect()
        torch.cuda.empty_cache()

    for items in tqdm(loader, desc=f"PQA aggregation {config.k_shots}-shot"):
        paths = items["img_path"]
        with torch.no_grad():
            features, patches = model.encode_image(items["img"].to(device), config.features_list, DPAM_layer=dpam)
            prompt_features, prompt_patches = prompt_association(*memory, items["cls_name"])
            raw, maps, scores, maxima, globals_, fixed, details = infer_batch(
                config, features, patches, prompt_features, prompt_patches, textual, visual, pq, learned_text,
                paths, args.effective_strength)
        masks = items["img_mask"][:, 0].clone()
        masks[masks > .5], masks[masks <= .5] = 1, 0
        if any(value.shape != masks.shape for value in maps.values()) or any(value.shape != (len(paths),) for value in scores.values()):
            raise ValueError("Unexpected map or score dimensions.")
        routing.append({"Batch": len(routing), "Sample Count": len(paths), "Strength": args.effective_strength,
                        "Original Dual Image Source": "baseline", "Weighted Dual Image Source": "weighted",
                        "Disabled Recovery": "passed" if args.effective_strength == 0 else "not_requested"})
        for index, category in enumerate(items["cls_name"]):
            path, label = paths[index], int(items["anomaly"][index])
            if processed >= len(source["manifest"]):
                raise ValueError("Unexpected extra test images.")
            expected = source["manifest"][processed]
            if (path_key(path) != path_key(expected["Image Path"]) or category != expected["Category"]
                    or label != int(expected["Sample Type"] == "Anomaly")):
                raise ValueError(f"Runtime identity/order/label mismatch: {path}")
            if current is None or current.category != category:
                if current is not None:
                    flush(current)
                    closed.add(current.category)
                if category in closed:
                    raise ValueError("Classes must be contiguous for bounded buffering.")
                current = CategoryBuffer(category)
            current.append(masks[index], label, maps, scores, index)
            row = {"Category": category, "Image Path": path, "Label": label,
                   **{f"{name} Global": float(value[index]) for name, value in globals_.items()},
                   "Fixed Global Mean": float(fixed[index])}
            for name in maps:
                row.update({f"{name} Map Max": float(maxima[name][index]), f"{name} Image Score": float(scores[name][index])})
            for condition in CONDITIONS:
                row.update({f"{condition} Output Image Score": float(scores[IMAGE_SOURCES[condition]][index]),
                            f"{condition} Pixel Source": PIXEL_SOURCES[condition], f"{condition} Image Source": IMAGE_SOURCES[condition]})
            score_rows.append(row)
            score_checks.append(compare_scores(row, recorded["scores"][processed], args.score_atol, args.score_rtol))
            if not score_checks[-1]["Passed"]:
                save_checks()
                raise RuntimeError(f"Original per-image components changed: {path}")
            weight_row = {"Category": category, "Image Path": path, "Fallback": bool(details["fallback"][index]),
                          "Effective Strength": args.effective_strength}
            for layer_index, layer in enumerate(config.features_list):
                weight_row.update({f"L{layer} Weight": float(details["weights"][index, layer_index]),
                                   f"L{layer} Agreement": float(details["agreement"][index, layer_index])})
            weights.append(weight_row)
            key = path_key(path)
            if key in selected:
                checks = save_sample(args, source, output, selected[key], items["img"][index], masks[index], raw, maps, index)
                map_checks.extend(checks)
                visited.add(key)
                if any(check["Status"] == "failed" for check in checks):
                    save_checks()
                    raise RuntimeError("Original FINAL map comparison failed.")
            processed += 1
        del features, patches, prompt_features, prompt_patches, raw, maps, scores, maxima, globals_, fixed, details
    if current is None or processed != len(source["manifest"]) or visited != set(selected):
        raise ValueError("Incomplete test or selected image coverage.")
    flush(current)
    combined = metric_tables(output, pixels, images)
    verify_metrics(combined)
    write_csv(output / "metrics_combined.csv", ["Category", "Condition", *FIELDS], combined)
    return {"status": "completed", "complete": True, "sample_count": processed, "category_count": len(closed) + 1,
            "conditions": list(CONDITIONS), "effective_strength": args.effective_strength,
            "controls": "passed_within_tolerance", "per_image_control_checks": len(score_checks),
            "visualization_samples": len(visited), "fallback_samples": sum(row["Fallback"] for row in weights),
            "baseline_map_checks": {state: sum(row["Status"] == state for row in map_checks)
                                    for state in ("passed", "failed", "unavailable")},
            "disabled_exact_recovery": "passed" if args.effective_strength == 0 else "not_requested"}


def run_worker(args):
    session = Path(args._worker_session).resolve()
    session.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    output = session / f"visa_{args.k_shots}shot_seed10"
    output.mkdir()
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    logger = logging.getLogger("pqa_layer_aggregation")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        source = audit_source(args.source_root / f"visa_{args.k_shots}shot_seed10", args.k_shots, PROJECT_ROOT)
        selection = select_samples([source], args.vis_per_type)
        recorded = audit_ablation(source, args.control_root / f"visa_{args.k_shots}shot_seed10")
        audit = read_json(session / "input_audit.json")
        original = next(row for row in audit["runs"] if row["shot"] == args.k_shots)
        control = next(row for row in audit["control_runs"] if row["shot"] == args.k_shots)
        if (source["hashes"] != original["source_file_sha256"] or recorded["hashes"] != control["hashes"]
                or selection["samples"] != audit["visualization_selection"]["samples"]
                or args.effective_strength != audit["effective_strength"]):
            raise ValueError("Inputs or effective aggregation changed after parent audit.")
        import torch
        import numpy as np
        import scipy
        write_json(output / "experiment_config.json", {
            **source["config"], "source_result_dir": source["source"], "control_result_dir": recorded["directory"],
            "result_dir": str(output), "project_root": str(PROJECT_ROOT), "conditions": list(CONDITIONS),
            "aggregation": "positive centered spatial correlation, uniform-regularized layer weights",
            "aggregation_strength": args.aggregation_strength, "disable_aggregation": args.disable_aggregation,
            "effective_strength": args.effective_strength, "agreement_grid": GRID_SIZE,
            "pixel_sources": PIXEL_SOURCES, "image_sources": IMAGE_SOURCES,
            "native_global_policy": "Original mean of VA/TA/PQA native globals; unchanged for all conditions",
            "weight_scope": "Independent per image; no categories, labels, cross-batch or fitted state",
            "score_policy": "Each family uses its WITH-TA map maximum for image scoring, including the family's dual condition",
            "num_workers": args.num_workers, "save_maps": args.save_maps, "save_visualizations": args.save_visualizations,
            "vis_per_class": args.vis_per_type, "visualization_sample_count": len(source["diagnostics"]),
            "maps_scope": "Selected samples only: four raw/evaluated pixel maps",
            "statistics_stage": "Per-image layer weights/agreements, original score parity and output sources",
            "visualization": "Fixed [0,1], input/GT/four unique outputs; no prediction normalization",
            "baseline_verification": "Original baseline metrics/maps and completed dual-output controls",
            "source_versions": source["config"].get("versions"),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
            "tolerances": {key: getattr(args, key) for key in ("baseline_metric_atol", "baseline_map_atol", "baseline_map_rtol", "score_atol", "score_rtol")},
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
    audit.update({"visualization_selection": select_samples(sources, args.vis_per_type),
                  "effective_strength": args.effective_strength, "control_runs": []})
    for source, shot in zip(sources, shots):
        recorded = audit_ablation(source, args.control_root / f"visa_{shot}shot_seed10")
        audit["control_runs"].append({"shot": shot, "directory": recorded["directory"], "hashes": recorded["hashes"]})
    session = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "visa_seed10")
    write_json(session / "input_audit.json", audit)
    write_json(session / "run_status.json", {"status": "running", "complete": False, "shots": shots})
    print(f"Results: {session}", flush=True)
    try:
        for position, shot in enumerate(shots):
            command = [sys.executable, str(Path(__file__).resolve()), "--k_shots", str(shot),
                       "--source_root", str(args.source_root), "--control_root", str(args.control_root),
                       "--_worker_session", str(session)]
            for name in ("num_workers", "vis_per_type", "aggregation_strength", "baseline_metric_atol", "baseline_map_atol", "baseline_map_rtol", "score_atol", "score_rtol"):
                command.extend([f"--{name}", str(getattr(args, name))])
            for name in ("save_maps", "disable_aggregation"):
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
                raise RuntimeError(f"{shot}-shot failed; remaining shots stopped (exit {code}).")
            status = read_json(session / f"visa_{shot}shot_seed10" / "run_status.json")
            if status.get("status") != "completed" or status.get("complete") is not True:
                raise RuntimeError(f"{shot}-shot incomplete; remaining shots stopped.")
            aggregate(session, shots[:position + 1])
        write_json(session / "run_status.json", {"status": "completed", "complete": True, "shots": shots})
    except BaseException as error:
        write_json(session / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error)})
        raise
    print(f"Completed. Summary: {session / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
