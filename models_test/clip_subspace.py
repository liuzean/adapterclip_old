"""Independent CLIP PCA residual vs original Baseline and dual output.

Project root: python models_test/clip_subspace.py
Default: sequential 1/2/4-shot VisA workers, seed=10, original recorded supports.
No extra image encoding, training, layer weighting, or score fusion for PCA.
"""

import argparse
import gc
import logging
import math
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from branch_ablation import infer_batch
from branch_ablation_utils import select_samples
from dual_output import save_sample as save_control_sample
from dual_output_utils import FIELDS, compare_scores
from fusion_ablation import aggregate, initialize
from fusion_ablation_utils import (
    audit_sequence, audit_source, compare_baseline, metric_tables, new_directory,
    path_key, project_path, read_json, save_panels, write_csv, write_json,
)
from clip_subspace_utils import (
    CONDITIONS, CategoryBuffer, audit_controls, candidate_outputs, check_reference_separation,
    compare_controls, fit_subspace, patch_descriptors,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/branch_diagnostics")
    parser.add_argument("--control_root", default="results/dual_output/visa_seed10")
    parser.add_argument("--k_shots", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument("--pca_ev", type=float, default=.99, help="Support-only retained variance; no test tuning.")
    parser.add_argument("--disable_pca", action="store_true", help="Skip fitting; candidate aliases both baseline outputs.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_per_type", type=int, default=1)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true", help="Only fixed visualization samples, not all test maps.")
    for name, default in (("baseline_metric_atol", .001), ("baseline_map_atol", 1e-6),
                          ("baseline_map_rtol", 1e-5), ("score_atol", 1e-6), ("score_rtol", 1e-5)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--_worker_session", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.pca_ev) or not 0 < args.pca_ev < 1:
        parser.error("pca_ev must be finite and strictly between 0 and 1.")
    if args.num_workers < 0 or args.vis_per_type < 1 or any(not math.isfinite(v) or v < 0 for v in (
            args.baseline_metric_atol, args.baseline_map_atol, args.baseline_map_rtol, args.score_atol, args.score_rtol)):
        parser.error("Counts/tolerances must be nonnegative and finite; vis_per_type must be positive.")
    if args._worker_session and args.k_shots is None:
        parser.error("Internal worker needs an explicit shot count.")
    for key in ("source_root", "control_root"):
        setattr(args, key, project_path(getattr(args, key), PROJECT_ROOT))
    return args


def save_sample(args, source, output, sample, image, mask, raw, maps, index):
    """Reuse old FINAL matrix parity and save selected PCA maps without rescaling."""
    import numpy as np
    control_args = SimpleNamespace(**{**vars(args), "disable_dual_output": False, "save_visualizations": False})
    checks = save_control_sample(control_args, source, output, sample, image, mask, raw, maps, index)
    relative = Path(sample["Category"]) / sample["Sample Type"] / sample["Sample Key"]
    names = {"baseline": "baseline", "dual_output": "no_TA", "clip_pca": "clip_pca"}
    raw_arrays = {name: raw[key][index].detach().cpu().numpy() for name, key in names.items()}
    arrays = {name: maps[key][index].numpy() for name, key in names.items()}
    if args.save_visualizations:
        save_panels(output / "visualizations" / relative.with_suffix(".png"), image, mask.numpy(),
                    raw_arrays, arrays, list(CONDITIONS), 3,
                    f"VisA {source['config']['k_shots']}-shot / {sample['Category']} / "
                    "PCA is residual/4, not probability; fixed display range")
    if args.save_maps:
        for stage, values in (("raw", raw_arrays), ("evaluated", arrays)):
            directory = output / "raw_maps" / relative / stage
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "clip_pca.npy", values["clip_pca"], allow_pickle=False)
    return checks


def run_inference(args, source, recorded, output, logger):
    """Reuse initialization, support encoding, adapters and evaluator unchanged."""
    import torch
    from test import prompt_association
    from tools import setup_seed
    from tqdm import tqdm
    setup_seed(source["config"]["seed"])
    config, model, textual, visual, pq, learned_text, memory, loader, dpam, device = initialize(
        source, output, args.num_workers, logger)
    states, fit_rows = {}, []
    # All fits finish BEFORE any query is decoded or passed to the model.
    if not args.disable_pca:
        fit_dir = output / "pca_models"
        fit_dir.mkdir()
        for category in source["classes"]:
            reference_layers = memory[1][category]
            if any(value.shape[0] != config.k_shots for value in reference_layers):
                raise ValueError("PCA support count differs from recorded shot budget.")
            logger.info("Fitting normal-only PCA: %s / %s references", category, config.k_shots)
            with torch.no_grad():
                descriptors = patch_descriptors(reference_layers)
                state = fit_subspace(descriptors.flatten(0, 1), args.pca_ev)
            states[category] = state
            torch.save(state, fit_dir / f"{category}.pt")
            fit_rows.append({"Category": category, "Reference Count": config.k_shots, **state["metadata"]})
            write_csv(output / "pca_fit.csv", list(fit_rows[0]), fit_rows)
            del descriptors, state
    selected = {path_key(row["Image Path"]): row for row in source["diagnostics"]}
    pixels, images, score_rows, score_checks, map_checks, routing = [], [], [], [], [], []
    buffer, processed, closed, visited = None, 0, set(), set()

    def save_checks():
        for filename, rows in (("image_scores.csv", score_rows), ("image_score_parity.csv", score_checks),
                               ("baseline_map_checks.csv", map_checks), ("routing_checks.csv", routing)):
            if rows:
                write_csv(output / filename, list(rows[0]), rows)

    def check_metrics(combined):
        baseline = compare_baseline(source, combined, args.baseline_metric_atol)
        controls = compare_controls(recorded, combined, args.baseline_metric_atol)
        write_csv(output / "baseline_reference_comparison.csv", list(baseline[0]), baseline)
        write_csv(output / "control_reference_comparison.csv", list(controls[0]), controls)
        if not all(row["Passed"] for row in [*baseline, *controls]):
            raise RuntimeError("Historical control metric mismatch; remaining shots stopped.")

    def flush(current):
        p_rows, i_rows = current.finish(device, logger)
        pixels.extend(p_rows)
        images.extend(i_rows)
        combined = metric_tables(output, pixels, images)
        save_checks()
        check_metrics([row for row in combined if row["Category"] != "Mean"])
        gc.collect()
        torch.cuda.empty_cache()

    for items in tqdm(loader, desc=f"CLIP subspace {config.k_shots}-shot"):
        paths, categories = items["img_path"], items["cls_name"]
        with torch.no_grad():
            features, patches = model.encode_image(items["img"].to(device), config.features_list, DPAM_layer=dpam)
            prompt_features, prompt_patches = prompt_association(*memory, categories)
            raw, maps, scores, maxima, globals_, fixed, extras, distances = infer_batch(
                config, features, patches, prompt_features, prompt_patches,
                textual, visual, pq, learned_text, paths, nullcontext())
            del extras, distances
            for values in (raw, maps, scores, maxima):
                values.pop("no_VA")
            raw["clip_pca"], maps["clip_pca"], scores["clip_pca"] = candidate_outputs(
                patches, categories, states, config.image_size, config.sigma,
                raw["baseline"], maps["baseline"], scores["baseline"], args.disable_pca)
        maps["dual_output"], scores["dual_output"] = maps["no_TA"], scores["baseline"]
        checks = {"Batch": len(routing), "Sample Count": len(paths),
                  "Dual Pixel Equal": torch.equal(maps["dual_output"], maps["no_TA"]),
                  "Dual Image Equal": torch.equal(scores["dual_output"], scores["baseline"]),
                  "Disable Requested": args.disable_pca,
                  "Disabled Recovery": (torch.equal(maps["clip_pca"], maps["baseline"])
                                        and torch.equal(scores["clip_pca"], scores["baseline"])) if args.disable_pca else "not_requested"}
        routing.append(checks)
        if not checks["Dual Pixel Equal"] or not checks["Dual Image Equal"] or checks["Disabled Recovery"] is False:
            save_checks()
            raise RuntimeError("Output routing/recovery failed.")
        masks = (items["img_mask"][:, 0] > .5).to(torch.uint8)
        for name in CONDITIONS:
            if maps[name].shape != masks.shape or scores[name].shape != (len(paths),):
                raise ValueError(f"Unexpected output dimensions: {name}")
        for index, category in enumerate(categories):
            path, label = paths[index], int(items["anomaly"][index])
            if processed >= len(source["manifest"]):
                raise ValueError("Extra query sample.")
            expected = source["manifest"][processed]
            if (path_key(path) != path_key(expected["Image Path"]) or category != expected["Category"]
                    or label != int(expected["Sample Type"] == "Anomaly")):
                raise ValueError(f"Query identity/order/label mismatch: {path}")
            if buffer is None or buffer.category != category:
                if buffer is not None:
                    flush(buffer)
                    closed.add(buffer.category)
                if category in closed:
                    raise ValueError("Classes must be contiguous for bounded buffering.")
                buffer = CategoryBuffer(category, args.disable_pca)
            buffer.append(masks[index], label, maps, scores, index)
            row = {"Category": category, "Image Path": path, "Label": label,
                   **{f"{name} Global": float(value[index]) for name, value in globals_.items()},
                   "Fixed Global Mean": float(fixed[index])}
            for name in ("baseline", "no_TA"):
                row.update({f"{name} Map Max": float(maxima[name][index]), f"{name} Image Score": float(scores[name][index])})
            row.update({"dual_output Image Score": float(scores["dual_output"][index]),
                        "clip_pca Image Score": float(scores["clip_pca"][index]),
                        "clip_pca Map Min": float(maps["clip_pca"][index].min()),
                        "clip_pca Map Max": float(maps["clip_pca"][index].max()),
                        "clip_pca Map Mean": float(maps["clip_pca"][index].mean())})
            score_rows.append(row)
            score_checks.append(compare_scores(row, recorded["scores"][processed], args.score_atol, args.score_rtol))
            if not score_checks[-1]["Passed"]:
                save_checks()
                raise RuntimeError(f"Historical control score mismatch: {path}")
            key = path_key(path)
            if key in selected:
                checks = save_sample(args, source, output, selected[key], items["img"][index], masks[index], raw, maps, index)
                map_checks.extend(checks)
                visited.add(key)
                if any(check["Status"] == "failed" for check in checks):
                    save_checks()
                    raise RuntimeError("Original FINAL map mismatch.")
            processed += 1
        del features, patches, prompt_features, prompt_patches, raw, maps, scores, maxima, globals_, fixed
    if buffer is None or processed != len(source["manifest"]) or visited != set(selected):
        raise ValueError("Incomplete query/visualization coverage.")
    flush(buffer)
    combined = metric_tables(output, pixels, images)
    check_metrics(combined)
    write_csv(output / "metrics_combined.csv", ["Category", "Condition", *FIELDS], combined)
    return {"status": "completed", "complete": True, "sample_count": processed, "category_count": len(closed) + 1,
            "conditions": list(CONDITIONS), "pca_fit_categories": len(states), "disable_pca": args.disable_pca,
            "visualization_sample_count": len(visited), "control_metrics": "passed_within_tolerance",
            "image_score_checks": len(score_checks), "baseline_map_checks": {
                state: sum(row["Status"] == state for row in map_checks) for state in ("passed", "failed", "unavailable")},
            "extra_target_images_for_fit": 0, "query_features_used_for_fit": False, "official_subspacead_reproduction": False}


def run_worker(args):
    session = Path(args._worker_session).resolve()
    session.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    output = session / f"visa_{args.k_shots}shot_seed10"
    output.mkdir()
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    logger = logging.getLogger("clip_subspace")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        source = audit_source(args.source_root / f"visa_{args.k_shots}shot_seed10", args.k_shots, PROJECT_ROOT)
        check_reference_separation(source)
        selection = select_samples([source], args.vis_per_type)
        recorded = audit_controls(source, args.control_root / f"visa_{args.k_shots}shot_seed10")
        audit = read_json(session / "input_audit.json")
        previous = next(row for row in audit["runs"] if row["shot"] == args.k_shots)
        control = next(row for row in audit["control_runs"] if row["shot"] == args.k_shots)
        if (source["hashes"] != previous["source_file_sha256"] or recorded["hashes"] != control["hashes"]
                or selection["samples"] != audit["selection"]["samples"]
                or audit["candidate"] != {"pca_ev": args.pca_ev, "disable_pca": args.disable_pca}):
            raise ValueError("Inputs/candidate configuration changed after parent audit.")
        import torch
        import numpy as np
        import scipy
        write_json(output / "experiment_config.json", {
            **source["config"], **{key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items() if not key.startswith("_")},
            "result_dir": str(output), "conditions": list(CONDITIONS),
            "source_result_dir": source["source"], "control_result_dir": recorded["directory"],
            "feature_policy": "equal mean of projected CLIP L6/L12/L18/L24 patches, remove CLS then L2 normalize",
            "fit_policy": "recorded normal TRAIN support cache only; no augmentation; CPU float64 covariance/eigh",
            "score_policy": "float32 squared PCA residual / 4; bilinear resize; original SciPy smoothing; image max",
            "candidate_output_policy": "independent residual, no fusion; aliases baseline when disabled",
            "control_policy": "baseline unchanged; dual pixel=no_TA, dual image=baseline",
            "pca_fit_precision": "float64", "pca_score_precision": "float32", "augmentation_count": 0,
            "save_maps_scope": "fixed selected samples only", "visualization_range": [0, 1],
            "extra_target_images_for_fit": 0, "official_subspacead_reproduction": False,
            "source_versions": source["config"].get("versions"),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        })
        (output / "reference_samples.txt").write_text("".join(f"{c}\t{p}\n" for c, p in source["reference_paths"]), encoding="utf-8")
        write_csv(output / "source_sample_manifest.csv", list(source["manifest"][0]), source["manifest"])
        write_csv(output / "visualization_samples.csv", list(selection["samples"][0]), selection["samples"])
        write_json(output / "run_status.json", run_inference(args, source, recorded, output, logger))
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error)})
        logger.exception("Run stopped; outputs may be incomplete.")
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
    for source in sources:
        check_reference_separation(source)
    audit = audit_sequence(sources)
    audit["selection"] = select_samples(sources, args.vis_per_type)
    audit["candidate"] = {"pca_ev": args.pca_ev, "disable_pca": args.disable_pca}
    audit["control_runs"] = []
    for source, shot in zip(sources, shots):
        recorded = audit_controls(source, args.control_root / f"visa_{shot}shot_seed10")
        audit["control_runs"].append({"shot": shot, "directory": recorded["directory"], "hashes": recorded["hashes"]})
    session = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "visa_seed10")
    write_json(session / "input_audit.json", audit)
    write_json(session / "run_status.json", {"status": "running", "complete": False, "shots": shots})
    print(f"Results: {session}", flush=True)
    try:
        for position, shot in enumerate(shots):
            command = [sys.executable, str(Path(__file__).resolve()), "--k_shots", str(shot), "--_worker_session", str(session)]
            for key in ("source_root", "control_root", "pca_ev", "num_workers", "vis_per_type",
                        "baseline_metric_atol", "baseline_map_atol", "baseline_map_rtol", "score_atol", "score_rtol"):
                command.extend([f"--{key}", str(getattr(args, key))])
            for key in ("save_maps", "disable_pca"):
                if getattr(args, key):
                    command.append(f"--{key}")
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
                raise RuntimeError(f"{shot}-shot failed ({code}); remaining shots stopped.")
            status = read_json(session / f"visa_{shot}shot_seed10" / "run_status.json")
            if status.get("complete") is not True or status.get("status") != "completed":
                raise RuntimeError("Worker incomplete; remaining shots stopped.")
            aggregate(session, shots[:position + 1])
        write_json(session / "run_status.json", {"status": "completed", "complete": True, "shots": shots})
    except BaseException as error:
        write_json(session / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error)})
        raise
    print(f"Completed. Summary: {session / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
