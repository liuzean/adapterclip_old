"""Manually run five existing experiments on five NEW datasets, stage by stage.

Project root: python tests/run_five_experiments.py
Order: experiment -> BTAD/MPDD/SDD/DTD/part_glass -> 1/2/4-shot (75 workers).
Every worker calls an existing experiment function, cleans up, then exits before
the next worker starts. This is an opt-in integration driver, not a pytest test.

--dry_run prints the full plan without inference. --datasets selects a subset
of the five new datasets. Seed/model/algorithm defaults come from the original
experiments; checkpoint/dataset root come from test_all.py. Selected matrices
are saved in stages 1-2 only; stages 3-5 keep no new matrices.
Results stay in results/<experiment>/<campaign>/<dataset>/<shot>shot_seed10;
the combined summary is under results/run_five_experiments/<campaign>.
Failure stops the campaign; repeated launches create a new campaign, not resume.
"""

import argparse
import gc
import hashlib
import importlib
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in (PROJECT_ROOT, PROJECT_ROOT / "models_test"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# test_all has standard-library imports and a guarded main; no inference here.
from test_all import CHECKPOINT_PATH, DATASET_ALIASES, DATASET_ROOT
from fusion_ablation_utils import (
    METRICS, audit_source, new_directory, path_key, read_csv, read_json, write_csv,
)

DATASETS = ("BTAD", "MPDD", "SDD", "DTD", "part_glass")
EXPERIMENTS = ("branch_diagnostics", "fusion_ablation", "branch_ablation", "dual_output", "pqa_layer_aggregation")
MAP_SAVING = {experiment: experiment in ("branch_diagnostics", "fusion_ablation") for experiment in EXPERIMENTS}
SHOTS = (1, 2, 4)
FIELDS = tuple(f"{level}-{metric}" for level in ("I", "P") for metric in METRICS)
PRETRAINED = "E:/projects/FSAD/AdaptCLIP/checkpoint/train_base/ViT-L-14-336px.pt"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dataset_root", default=str(DATASET_ROOT))
    parser.add_argument("--checkpoint_path", default=str(CHECKPOINT_PATH))
    parser.add_argument("--pretrained_model", default=PRETRAINED)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Check metadata and print all worker commands; no model or results written.")
    parser.add_argument("--_campaign", help=argparse.SUPPRESS)
    parser.add_argument("--_experiment", choices=EXPERIMENTS, help=argparse.SUPPRESS)
    parser.add_argument("--_dataset", choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--_shot", type=int, choices=SHOTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or len(set(args.datasets)) != len(args.datasets):
        parser.error("num_workers must be nonnegative; dataset names must be unique.")
    for name in ("dataset_root", "checkpoint_path", "pretrained_model"):
        setattr(args, name, resolve_path(getattr(args, name)))
    internal = (args._campaign, args._experiment, args._dataset, args._shot)
    if any(value is not None for value in internal) and not all(value is not None for value in internal):
        parser.error("All internal worker arguments are required together.")
    return args


def metadata_audit(folder):
    """Read JSON/path identities only; reject missing files or an invalid protocol.

    All selected datasets need normal TRAIN supports and >=3 normal/anomalous
    test images per class for the unchanged diagnostic sample budgets.
    No image pixels, masks or tensors are decoded by this parent-side audit.
    """
    metadata_path = folder / "meta.json"
    metadata = read_json(metadata_path)
    train, test = metadata.get("train", {}), metadata.get("test", {})
    if not test or set(train) != set(test):
        raise ValueError(f"Normal TRAIN and TEST class sets must match: {folder}")
    test_keys, train_keys, counts = set(), set(), {}
    for category, rows in train.items():
        if not rows:
            raise ValueError(f"No support candidates: {folder}/{category}")
        for row in rows:
            if row["cls_name"] != category or int(row["anomaly"]) != 0:
                raise ValueError(f"Reference candidates must all be normal TRAIN: {folder}/{category}")
            path = (folder / row["img_path"]).resolve()
            path.relative_to(folder)
            if not path.is_file():
                raise FileNotFoundError(path)
            train_keys.add(path_key(path))
    for category, rows in test.items():
        labels = Counter()
        for row in rows:
            label = int(row["anomaly"])
            if row["cls_name"] != category or label not in (0, 1):
                raise ValueError(f"Invalid TEST category/label: {folder}/{category}")
            labels[label] += 1
            path = (folder / row["img_path"]).resolve()
            path.relative_to(folder)
            key = path_key(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if key in test_keys:
                raise ValueError(f"Duplicate query image: {path}")
            test_keys.add(key)
            if label:
                mask = (folder / row["mask_path"]).resolve()
                mask.relative_to(folder)
                if not mask.is_file():
                    raise FileNotFoundError(mask)
        if labels[0] < 3 or labels[1] < 3:
            raise ValueError(f"Need >=3 normal and anomalous queries for diagnostics: {folder}/{category}")
        counts[category] = {"normal": labels[0], "anomaly": labels[1], "train": len(train[category])}
    if train_keys & test_keys:
        raise ValueError(f"TRAIN/TEST image paths overlap: {folder}")
    return {"root": str(folder), "classes": list(test), "counts": counts,
            "sample_count": len(test_keys), "meta_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}


def round_dir(campaign, experiment, dataset, shot):
    """Each experiment keeps its own results/<entry_stem>/ root."""
    return PROJECT_ROOT / "results" / experiment / campaign.name / dataset / f"{shot}shot_seed10"


def worker_command(args, campaign, experiment, dataset, shot):
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--_campaign", str(campaign), "--_experiment", experiment,
               "--_dataset", dataset, "--_shot", str(shot), "--datasets", *args.datasets,
               "--dataset_root", str(args.dataset_root), "--checkpoint_path", str(args.checkpoint_path),
               "--pretrained_model", str(args.pretrained_model), "--num_workers", str(args.num_workers)]
    if not args.save_visualizations:
        command.append("--no-save_visualizations")
    return command


def run_and_wait(command):
    """Reap every worker before continuing; Ctrl+C never leaves the next queued job running."""
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
        raise RuntimeError(f"Worker failed with exit code {code}; remaining rounds stopped.")


def summarize(campaign, completed):
    """Merge existing CSVs only; never import an evaluator in the parent process."""
    rows = []
    for experiment, dataset, shot in completed:
        output = round_dir(campaign, experiment, dataset, shot)
        for row in read_csv(output / "metrics_combined.csv"):
            rows.append({"Experiment": experiment, "Dataset": dataset, "Shot": shot, **row})
    columns = ["Experiment", "Dataset", "Shot", "Category", "Condition", *FIELDS]
    write_csv(campaign / "metrics_summary.csv", columns, [row for row in rows if row["Category"] == "Mean"])
    write_csv(campaign / "metrics_by_category.csv", columns, [row for row in rows if row["Category"] != "Mean"])


def diagnostic_metric_table(output):
    """Reformat stage-1 branch results, without computing any new metric."""
    pixels = read_csv(output / "pixel_metrics.csv")
    images = read_csv(output / "image_metrics.csv")
    combined = []
    for pixel in pixels:
        branch, category = pixel["Branch"], pixel["Category"]
        score_source = "official_final" if branch == "FINAL" else "map_max" if branch == "ALIGN" else "native_global"
        matches = [row for row in images if (row["Category"], row["Branch"], row["Score Source"]) ==
                   (category, branch, score_source)]
        if len(matches) != 1:
            raise ValueError("Missing/duplicated primary image score metrics.")
        combined.append({"Category": category, "Condition": branch,
                         **{f"P-{m}": pixel[m] for m in METRICS},
                         **{f"I-{m}": matches[0][m] for m in METRICS}})
    write_csv(output / "metrics_combined.csv", ["Category", "Condition", *FIELDS], combined)


def recorded_source(campaign, dataset, shot):
    """Audit this campaign's stage-1 source using metadata-defined classes."""
    canonical = DATASET_ALIASES.get(dataset.casefold(), dataset)
    source = audit_source(round_dir(campaign, "branch_diagnostics", dataset, shot), shot,
                          PROJECT_ROOT, expected_dataset=canonical)
    return source


def write_diagnostic_config(output, args):
    import torch
    import scipy
    import numpy as np
    write_json(output / "experiment_config.json", {
        **vars(args), "project_root": str(PROJECT_ROOT), "result_dir": str(output),
        "metrics": list(FIELDS), "map_max_stage": "after Gaussian smoothing",
        "metric_scale": "0-100 percent; N/A for single-label target sets",
        "statistics_stage": "before Gaussian smoothing, pooled pixels per category and sample type",
        "ap_definition": "Unmodified tools.Evaluator AUPR (trapezoidal PR area)",
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "evaluation_scope": "full_dataset", "driver": "tests/run_five_experiments.py",
    })


def run_diagnostics(args, campaign, output, logger):
    """Use the original parser defaults and run(), overriding the dataset identity.

    Its CLI is VisA-only. This integration driver builds a validated Namespace
    for run(), which already uses the generic Dataset/PromptDataset classes.
    No monkey-patching, copied model forward, or fake VisA metadata is involved.
    """
    import branch_diagnostics as module
    options = module.parse_args([
        "--k_shots", str(args._shot), "--test_data_path", str(args.dataset_root / args._dataset),
        "--pretrained_model", str(args.pretrained_model), "--checkpoint_path", str(args.checkpoint_path),
        "--num_workers", str(args.num_workers), "--save_maps",
        *([] if args.save_visualizations else ["--no-save_visualizations"]),
    ])
    options.dataset = DATASET_ALIASES.get(args._dataset.casefold(), args._dataset)
    write_diagnostic_config(output, options)
    status = module.run(options, output, logger)
    # run()'s original status string and historical-log lookup were VisA-only.
    # The current dataset is explicit in this driver's status/configuration.
    status.update(complete=True, evaluation_scope="full_dataset", dataset=options.dataset)
    write_json(output / "baseline_comparison.json", {
        "status": "new_dataset_control_generated", "dataset": options.dataset,
        "note": "No VisA historical result used. Later stages compare this campaign's full-precision outputs."})
    diagnostic_metric_table(output)
    return status


def run_ablation(args, campaign, output, logger):
    """Call the original inference function for stages 2-5, with audited inputs."""
    from branch_ablation_utils import select_samples
    from dual_output_utils import audit_ablation
    module = importlib.import_module(args._experiment)
    options = module.parse_args(["--k_shots", str(args._shot), "--num_workers", str(args.num_workers)])
    options.save_visualizations = args.save_visualizations
    options.save_maps = MAP_SAVING[args._experiment]
    source = recorded_source(campaign, args._dataset, args._shot)
    if args._experiment == "branch_ablation":
        selection = select_samples([source], options.diagnostic_per_type)
    elif args._experiment in ("dual_output", "pqa_layer_aggregation"):
        selection = select_samples([source], options.vis_per_type)
    else:
        selection = {"policy": "First class in metadata; first 3 anomalous and first normal image",
                     "samples": source["diagnostics"]}
    recorded = None
    if args._experiment == "dual_output":
        control_dir = round_dir(campaign, "branch_ablation", args._dataset, args._shot)
        recorded = audit_ablation(source, control_dir)
    elif args._experiment == "pqa_layer_aggregation":
        control_dir = round_dir(campaign, "dual_output", args._dataset, args._shot)
        recorded = audit_ablation(source, control_dir)
    config = {**source["config"], "result_dir": str(output), "source_result_dir": source["source"],
              "conditions": list(module.CONDITIONS), "num_workers": options.num_workers,
              "save_maps": options.save_maps, "save_visualizations": options.save_visualizations,
              "evaluation_scope": "full_dataset", "driver": "tests/run_five_experiments.py"}
    if args._experiment == "dual_output":
        config.update(disable_dual_output=options.disable_dual_output, pixel_source="no_TA", image_source="baseline")
    if args._experiment == "pqa_layer_aggregation":
        config.update(aggregation_strength=options.aggregation_strength, effective_strength=options.effective_strength,
                      disable_aggregation=options.disable_aggregation, agreement_grid=module.GRID_SIZE,
                      pixel_sources=module.PIXEL_SOURCES, image_sources=module.IMAGE_SOURCES)
    if recorded is not None:
        config["control_result_dir"] = recorded["directory"]
    write_json(output / "experiment_config.json", config)
    write_json(output / "input_audit.json", {"source_file_sha256": source["hashes"],
               "control_file_sha256": recorded["hashes"] if recorded else None,
               "reference_selection": "Exact stage-1 normal TRAIN list, order and original encoding batch size"})
    write_json(output / "diagnostic_config.json", {**selection, "layers": source["config"]["features_list"],
               "scope": "Fixed selected examples, not whole-dataset layer performance"})
    (output / "reference_samples.txt").write_text("".join(f"{c}\t{p}\n" for c, p in source["reference_paths"]), encoding="utf-8")
    write_csv(output / "source_sample_manifest.csv", list(source["manifest"][0]), source["manifest"])
    sample_rows = [{key: row[key] for key in ("Category", "Sample Type", "Image Path", "Sample Key")}
                   for row in source["diagnostics"]]
    write_csv(output / "diagnostic_samples.csv", list(sample_rows[0]), sample_rows)
    if recorded is None:
        return module.run_inference(options, source, output, logger)
    return module.run_inference(options, source, recorded, output, logger)


def run_worker(args):
    campaign = resolve_path(args._campaign)
    campaign.relative_to(PROJECT_ROOT / "results" / Path(__file__).stem)
    saved = read_json(campaign / "campaign_config.json")
    for field in ("dataset_root", "checkpoint_path", "pretrained_model", "num_workers", "save_visualizations"):
        expected, actual = saved[field], getattr(args, field)
        if str(expected) != str(actual):
            raise ValueError(f"Worker configuration differs from campaign: {field}")
    if saved["map_saving"] != MAP_SAVING:
        raise ValueError("Matrix saving policy changed after campaign creation.")
    audit = metadata_audit(args.dataset_root / args._dataset)
    if audit != saved["metadata"][args._dataset]:
        raise ValueError("Dataset metadata changed after preflight.")
    output = round_dir(campaign, args._experiment, args._dataset, args._shot)
    output.mkdir(parents=True)  # refuse to overwrite even incomplete previous outputs
    logger = logging.getLogger("five_experiments")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    write_json(output / "run_status.json", {"status": "running", "complete": False})
    try:
        if args._experiment == "branch_diagnostics":
            status = run_diagnostics(args, campaign, output, logger)
        else:
            status = run_ablation(args, campaign, output, logger)
        if (status.get("status") != "completed" or status.get("sample_count") != audit["sample_count"]
                or status.get("category_count") != len(audit["classes"])):
            raise ValueError("Incomplete full-dataset coverage.")
        status.update(complete=True, dataset=args._dataset, k_shots=args._shot, evaluation_scope="full_dataset")
        write_json(output / "run_status.json", status)
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "complete": False, "error": str(error)})
        logger.exception("Round stopped; results may be partial.")
        raise
    finally:
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            sys.modules["torch"].cuda.empty_cache()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    # No model/data objects remain in this frame. OS process exit releases any
    # native allocator/CUDA context state; parent waits for that exit.


def main(argv=None):
    args = parse_args(argv)
    if args._campaign:
        run_worker(args)
        return
    for path in (args.checkpoint_path, args.pretrained_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = {name: metadata_audit(args.dataset_root / name) for name in args.datasets}
    jobs = [(experiment, dataset, shot) for experiment in EXPERIMENTS for dataset in args.datasets for shot in SHOTS]
    print(f"Experiments: {list(EXPERIMENTS)}\nDatasets: {args.datasets}\nShots: {SHOTS}\nWorkers: {len(jobs)}", flush=True)
    if args.dry_run:
        placeholder = PROJECT_ROOT / "results" / Path(__file__).stem / "PREVIEW_ONLY"
        for index, (experiment, dataset, shot) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {experiment} / {dataset} / {shot}-shot\n"
                  + subprocess.list2cmdline(worker_command(args, placeholder, experiment, dataset, shot)), flush=True)
        return
    campaign = new_directory(PROJECT_ROOT / "results" / Path(__file__).stem, "multi_dataset_seed10")
    write_json(campaign / "campaign_config.json", {**{k: v for k, v in vars(args).items() if not k.startswith("_")},
               "experiments": EXPERIMENTS, "shots": SHOTS, "seed": 10, "metadata": metadata,
               "map_saving": MAP_SAVING,
               "protocol": "original model parameters and six metrics; identical support list within each dataset/shot"})
    write_json(campaign / "execution_plan.json", [{"Experiment": e, "Dataset": d, "Shot": s,
               "Output": str(round_dir(campaign, e, d, s))} for e, d, s in jobs])
    completed = []
    write_json(campaign / "run_status.json", {"status": "running", "complete": False, "completed_rounds": 0, "total_rounds": len(jobs)})
    print(f"Campaign: {campaign}", flush=True)
    try:
        for index, (experiment, dataset, shot) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {experiment} / {dataset} / {shot}-shot", flush=True)
            write_json(campaign / "run_status.json", {"status": "running", "complete": False,
                       "completed_rounds": len(completed), "total_rounds": len(jobs),
                       "current_round": [experiment, dataset, shot]})
            run_and_wait(worker_command(args, campaign, experiment, dataset, shot))
            output = round_dir(campaign, experiment, dataset, shot)
            status = read_json(output / "run_status.json")
            if status.get("complete") is not True or status.get("status") != "completed":
                raise RuntimeError(f"Worker did not report completion: {output}")
            completed.append((experiment, dataset, shot))
            summarize(campaign, completed)
            write_json(campaign / "run_status.json", {"status": "running", "complete": False,
                       "completed_rounds": len(completed), "total_rounds": len(jobs), "last_completed": completed[-1]})
            print("Worker exited; its CPU/GPU allocations and CUDA context released before next shot.", flush=True)
        write_json(campaign / "run_status.json", {"status": "completed", "complete": True,
                   "completed_rounds": len(completed), "total_rounds": len(jobs)})
    except BaseException as error:
        write_json(campaign / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "complete": False, "completed_rounds": len(completed), "total_rounds": len(jobs),
                   "failed_round": jobs[len(completed)] if len(completed) < len(jobs) else None, "error": str(error)})
        raise
    print(f"Completed. Six-metric summary: {campaign / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
