"""Run the unchanged CLIP PCA candidate on five datasets, 1/2/4-shot/seed=10.

Project root: python models_test/clip_subspace_multidataset.py
Order: dataset -> 1/2/4-shot, each round in a fresh process (15 workers).
No VisA rerun, layer weighting, or PCA fusion is requested.
"""

import argparse
import gc
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import clip_subspace
from branch_ablation_utils import select_samples
from clip_subspace_utils import CONDITIONS, audit_controls, check_reference_separation
from dual_output_utils import FIELDS
from fusion_ablation_utils import (
    MODEL_KEYS, audit_source, new_directory, path_key, project_path, read_csv,
    read_json, write_csv, write_json,
)

DATASETS = ("BTAD", "MPDD", "SDD", "DTD", "part_glass")
SHOTS = (1, 2, 4)
DATASET_IDS = {"BTAD": "btad", "MPDD": "mpdd", "SDD": "SDD", "DTD": "DTD", "part_glass": "part_glass"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/branch_diagnostics/multi_dataset_seed10")
    parser.add_argument("--control_root", default="results/dual_output/multi_dataset_seed10")
    parser.add_argument("--original_root", default="results", help="Original dataset/shot test logs and prompts.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true", help="Save fixed examples only; off by default.")
    parser.add_argument("--dry_run", action="store_true", help="Read/audit inputs and print plan; no inference/output.")
    parser.add_argument("--_session", help=argparse.SUPPRESS)
    parser.add_argument("--_dataset", choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--_shot", type=int, choices=SHOTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("num_workers must be nonnegative.")
    worker_fields = (args._session is not None, args._dataset is not None, args._shot is not None)
    if any(worker_fields) and not all(worker_fields):
        parser.error("All three internal worker arguments are required together.")
    for key in ("source_root", "control_root", "original_root"):
        setattr(args, key, project_path(getattr(args, key), PROJECT_ROOT))
    return args


def original_audit(args, dataset, source):
    """Check the original matching-shot supports and rounded FINAL log.

    Original logs retain one decimal place. Runtime parity uses the existing
    full-precision diagnostic/control CSVs, not these rounded numbers.
    """
    shot = source["config"]["k_shots"]
    directory = args.original_root / dataset / str(shot)
    prefix = f"{DATASET_IDS[dataset]}_10seed_{shot}shot_test"
    prompts, log = directory / f"{prefix}_prompts.txt", directory / f"{prefix}_log.txt"
    root = Path(source["config"]["test_data_path"])
    original = [path_key(root / line.strip()) for line in prompts.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
    if original != [path_key(path) for _, path in source["reference_paths"]]:
        raise ValueError(f"Original normal references/order differ: {dataset}")
    rows = {}
    for line in log.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 7:
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        category = "Mean" if parts[0] == "Avg" else parts[0]
        if category in rows:
            raise ValueError(f"Duplicate original log category: {dataset}/{category}")
        rows[category] = values
    expected = set(source["classes"]) | ({"Mean"} if len(source["classes"]) > 1 else set())
    if set(rows) != expected:
        raise ValueError(f"Missing/extra original log categories: {dataset}")
    for category, values in rows.items():
        for field, value in zip(FIELDS, values):
            # Also rejects NaN/Inf in original logs through the positive bound.
            if not abs(value - source["metrics"][(category, field)]) <= .05001:
                raise ValueError(f"Original rounded Baseline differs: {dataset}/{category}/{field}")
    return {"directory": str(directory), "rounded_metric_cells": len(rows) * len(FIELDS),
            "log_rounding_tolerance_pp": .05001,
            "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (prompts, log)}}


def prepare(args, dataset, shot):
    """Audit controls/supports, select fixed examples, and freeze input hashes."""
    source = audit_source(args.source_root / dataset / f"{shot}shot_seed10", shot, PROJECT_ROOT,
                          expected_dataset=DATASET_IDS[dataset])
    check_reference_separation(source)
    selection = select_samples([source], 1)
    recorded = audit_controls(source, args.control_root / dataset / f"{shot}shot_seed10")
    metadata = Path(source["config"]["test_data_path"]) / "meta.json"
    audit = {"source_file_sha256": source["hashes"], "control_file_sha256": recorded["hashes"],
             "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
             "original_baseline": original_audit(args, dataset, source), "selection": selection,
             "source_result_dir": source["source"], "control_result_dir": recorded["directory"]}
    return source, recorded, audit


def worker_command(args, session, dataset, shot):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_session", str(session),
               "--_dataset", dataset, "--_shot", str(shot), "--num_workers", str(args.num_workers)]
    for key in ("source_root", "control_root", "original_root"):
        command.extend([f"--{key}", str(getattr(args, key))])
    if args.save_maps:
        command.append("--save_maps")
    if not args.save_visualizations:
        command.append("--no-save_visualizations")
    return command


def run_worker(args):
    session = Path(args._session).resolve()
    session.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    saved = read_json(session / "campaign_config.json")
    actual = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if not key.startswith("_")}
    if saved["options"] != actual:
        raise ValueError("Worker options changed after parent audit.")
    if saved["datasets"] != list(DATASETS) or saved["shots"] != list(SHOTS) or saved["seed"] != 10:
        raise ValueError("Dataset/shot plan changed after parent audit.")
    output = session / args._dataset / f"{args._shot}shot_seed10"
    output.mkdir(parents=True)  # Never reuse/overwrite an existing dataset round.
    logger = logging.getLogger("clip_subspace_multidataset")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    try:
        source, recorded, audit = prepare(args, args._dataset, args._shot)
        if audit != read_json(session / "input_audit.json")[args._dataset][str(args._shot)]:
            raise ValueError("Historical controls, metadata, or fixed examples changed after parent audit.")
        options = clip_subspace.parse_args(["--k_shots", str(args._shot), "--pca_ev", "0.99", "--vis_per_type", "1",
                                           "--num_workers", str(args.num_workers)])
        options.save_maps, options.save_visualizations = args.save_maps, args.save_visualizations
        import numpy as np
        import scipy
        import torch
        write_json(output / "experiment_config.json", {
            **source["config"], "result_dir": str(output), "driver": str(Path(__file__).resolve()),
            "source_result_dir": source["source"], "control_result_dir": recorded["directory"],
            "original_result_dir": audit["original_baseline"]["directory"],
            "pca_ev": options.pca_ev, "disable_pca": options.disable_pca,
            "conditions": list(CONDITIONS), "num_workers": options.num_workers,
            "vis_per_type": options.vis_per_type, "save_maps": options.save_maps,
            "save_visualizations": options.save_visualizations,
            "tolerances": {key: value for key, value in vars(options).items()
                           if key.endswith(("_atol", "_rtol"))},
            "feature_policy": "projected CLIP L6/L12/L18/L24; remove CLS; equal mean; L2 normalize",
            "fit_policy": "recorded normal TRAIN supports only; CPU float64 PCA; no augmentation",
            "score_policy": "float32 squared orthogonal residual / 4; original resize/sigma; image max",
            "candidate_output_policy": "independent residual, no fusion or layer weighting",
            "control_policy": "baseline unchanged; dual pixel=no_TA, dual image=baseline",
            "pca_fit_precision": "float64", "pca_score_precision": "float32", "augmentation_count": 0,
            "extra_target_images_for_fit": 0, "official_subspacead_reproduction": False,
            "save_maps_scope": "fixed selected examples only", "visualization_range": [0, 1],
            "source_versions": source["config"].get("versions"),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        })
        write_json(output / "input_audit.json", audit)
        (output / "reference_samples.txt").write_text(
            "".join(f"{category}\t{path}\n" for category, path in source["reference_paths"]), encoding="utf-8")
        write_csv(output / "source_sample_manifest.csv", list(source["manifest"][0]), source["manifest"])
        write_csv(output / "visualization_samples.csv", list(audit["selection"]["samples"][0]),
                  audit["selection"]["samples"])
        status = clip_subspace.run_inference(options, source, recorded, output, logger)
        if (status.get("complete") is not True or status.get("status") != "completed"
                or status.get("sample_count") != len(source["manifest"])
                or status.get("category_count") != len(source["classes"])):
            raise RuntimeError("Incomplete full-dataset PCA evaluation.")
        status.update(dataset=DATASET_IDS[args._dataset], k_shots=args._shot, seed=10, evaluation_scope="full_dataset")
        write_json(output / "run_status.json", status)
    except BaseException as error:
        write_json(output / "run_status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "complete": False, "error": str(error)})
        logger.exception("Dataset stopped; results may be incomplete.")
        raise
    finally:
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            sys.modules["torch"].cuda.empty_cache()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def summarize(session, completed):
    """Merge all six metrics and baseline differences without loading arrays."""
    rows, differences = [], []
    for dataset, shot in completed:
        combined = read_csv(session / dataset / f"{shot}shot_seed10" / "metrics_combined.csv")
        baselines = {row["Category"]: row for row in combined if row["Condition"] == "baseline"}
        for row in combined:
            rows.append({"Dataset": dataset, "Shot": shot, **row})
            if row["Condition"] != "baseline":
                differences.append({"Dataset": dataset, "Shot": shot, "Category": row["Category"],
                    "Condition": row["Condition"], "Comparator": "baseline",
                    **{field: float(row[field]) - float(baselines[row["Category"]][field]) for field in FIELDS}})
    fields = ["Dataset", "Shot", "Category", "Condition", *FIELDS]
    write_csv(session / "metrics_summary.csv", fields, [row for row in rows if row["Category"] == "Mean"])
    write_csv(session / "metrics_by_category.csv", fields, [row for row in rows if row["Category"] != "Mean"])
    write_csv(session / "metrics_vs_baseline.csv",
              ["Dataset", "Shot", "Category", "Condition", "Comparator", *FIELDS], differences)


def main(argv=None):
    args = parse_args(argv)
    if args._session:
        run_worker(args)
        return
    inputs, first = {}, None
    for dataset in DATASETS:
        inputs[dataset] = {}
        for shot in SHOTS:
            source, _, inputs[dataset][str(shot)] = prepare(args, dataset, shot)
            common = {key: source["config"][key] for key in MODEL_KEYS if key not in ("dataset", "test_data_path")}
            if first is not None and common != first:
                raise ValueError("Model/protocol settings differ across datasets or shots.")
            first = common
    jobs = [(dataset, shot) for dataset in DATASETS for shot in SHOTS]
    print(f"Datasets: {DATASETS}\nShots: {SHOTS}; seed: 10; PCA EV: 0.99; workers: {len(jobs)}", flush=True)
    if args.dry_run:
        for dataset, shot in jobs:
            print(f"{dataset} / {shot}-shot: {inputs[dataset][str(shot)]['source_result_dir']} -> independent CLIP PCA",
                  flush=True)
        return
    session = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "five_datasets_seed10")
    write_json(session / "input_audit.json", inputs)
    write_json(session / "campaign_config.json", {
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if not key.startswith("_")},
        "datasets": list(DATASETS), "shots": list(SHOTS), "seed": 10, "pca_ev": .99,
        "conditions": list(CONDITIONS), "candidate": "unchanged clip_subspace.run_inference"})
    write_json(session / "execution_plan.json", [{"Dataset": d, "Shot": s,
               "Output": str(session / d / f"{s}shot_seed10")} for d, s in jobs])
    completed = []
    current_round = None
    try:
        for index, (dataset, shot) in enumerate(jobs, 1):
            current_round = [dataset, shot]
            write_json(session / "run_status.json", {"status": "running", "complete": False,
                       "completed_rounds": len(completed), "total_rounds": len(jobs), "current_round": current_round})
            print(f"[{index}/{len(jobs)}] {dataset} / {shot}-shot\nResults: {session}", flush=True)
            with subprocess.Popen(worker_command(args, session, dataset, shot), cwd=str(PROJECT_ROOT)) as child:
                try:
                    code = child.wait()
                except KeyboardInterrupt:
                    try:
                        child.wait(timeout=15)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        if child.poll() is None:
                            child.terminate()
                        child.wait()
                    raise
            if code:
                raise RuntimeError(f"{dataset} / {shot}-shot worker failed ({code}); remaining rounds stopped.")
            status = read_json(session / dataset / f"{shot}shot_seed10" / "run_status.json")
            if status.get("complete") is not True or status.get("status") != "completed":
                raise RuntimeError(f"Incomplete worker: {dataset} / {shot}-shot")
            completed.append((dataset, shot))
            summarize(session, completed)
            print("Worker exited; its CPU/GPU resources released before the next round.", flush=True)
        write_json(session / "run_status.json", {"status": "completed", "complete": True,
                   "completed_rounds": len(completed), "total_rounds": len(jobs), "completed_jobs": completed})
    except BaseException as error:
        write_json(session / "run_status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "complete": False,
            "completed_rounds": len(completed), "total_rounds": len(jobs), "completed_jobs": completed,
            "failed_round": current_round, "error": str(error)})
        raise
    print(f"Completed. Six-metric summary: {session / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
