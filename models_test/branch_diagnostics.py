"""VisA branch diagnostics; default: sequential isolated 1/2/4-shot runs."""

import argparse
import gc
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args(argv=None):
    """Keep model defaults aligned with test.py; diagnostics always enable three adapters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["visa"], default="visa")
    parser.add_argument("--test_data_path", default=r"E:\projects\datasets\Visa")
    parser.add_argument("--pretrained_model", default=r"E:/projects/FSAD/AdaptCLIP/checkpoint/train_base/ViT-L-14-336px.pt")
    parser.add_argument("--checkpoint_path", default="checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--k_shots", type=int, choices=[1, 2, 4], default=None,
                        help="Run only this shot count; omitted = sequential 1, 2, 4-shot runs.")
    parser.add_argument("--fusion_type", choices=["average_mean", "harmonic_mean"], default="average_mean")
    parser.add_argument("--vl_reduction", type=int, default=4)
    parser.add_argument("--pq_mid_dim", type=int, default=128)
    parser.add_argument("--pq_context", action=argparse.BooleanOptionalAction, default=True,
                        help="Context enabled by default; match the checkpoint's training configuration.")
    for name in ("visual_learner", "textual_learner", "pq_learner"):
        parser.add_argument(f"--{name}", action="store_true", default=True, help="Enabled by default; required for five-branch diagnostics.")
    parser.add_argument("--vis_per_class", type=int, default=3, help="Number PER label (normal/anomaly), also used by selected map saving.")
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_maps", action="store_true", help="Save pre/post-smoothing .npy maps; default scope is selected samples only.")
    parser.add_argument("--maps_scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--smoke_per_type", type=int, default=0,
                        help="Smoke run only: first N normal and N anomalous samples/category; 0 = full evaluation.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.image_size < 1 or args.sigma < 0 or args.vis_per_class < 0 or args.num_workers < 0 or args.smoke_per_type < 0:
        parser.error("Invalid size, count, sigma or worker count.")
    if len(args.features_list) != 4:
        parser.error("Five-branch diagnosis requires the original four feature layers.")
    for name in ("test_data_path", "checkpoint_path", "pretrained_model"):
        path = Path(getattr(args, name)).expanduser()
        setattr(args, name, str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()))
    return args


def initialize(args, run_dir, logger):
    """Reuse test.py helpers and match its model/dataset/adapter initialization order.

    PromptDataset selects references BEFORE adapter initialization, exactly as test.py.
    Keep the original loaders (including their RNG effects) and original transforms.
    """
    import torch
    import adaptcliplib
    from adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter
    from dataset import Dataset, PromptDataset
    from test import build_prompt_memory, resolve_pretrained_model_type
    from tools import get_transform

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The stable TextualAdapter.prepare_static_text_feature calls .cuda() directly.
    if device != "cuda":
        raise RuntimeError("The original static text encoder requires CUDA; no CPU substitute is used.")
    model_type = resolve_pretrained_model_type(args.pretrained_model)
    if model_type == "ViT-L/14@336px":
        patch_size, input_dim, dpam_layer = 14, 768, 20
    elif model_type == "VITB16_PLUS_240":
        patch_size, input_dim, dpam_layer = 16, 640, 10
    else:
        raise ValueError(f"Unsupported local pretrained model: {args.pretrained_model}")
    if args.image_size % patch_size:
        raise ValueError(f"image_size must be divisible by patch_size={patch_size}")
    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=dpam_layer)
    preprocess, target_transform = get_transform(image_size=args.image_size)
    dataset_args = dict(root=args.test_data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset, k_shots=args.k_shots, save_dir=str(run_dir), mode="test", seed=args.seed)
    prompt_data = PromptDataset(**dataset_args)
    test_data = Dataset(**dataset_args)
    references = [{"category": row["cls_name"], "relative_path": row["img_path"],
                   "absolute_path": str(Path(args.test_data_path) / row["img_path"])} for row in prompt_data.prompt_data_all]
    if any(row["anomaly"] != 0 for row in prompt_data.prompt_data_all):
        raise ValueError("Original reference sampler selected an anomalous sample; inspect dataset metadata.")
    (run_dir / "reference_samples.txt").write_text(
        "".join(f"{row['category']}\t{row['absolute_path']}\n" for row in references), encoding="utf-8")
    prompt_loader = torch.utils.data.DataLoader(prompt_data, batch_size=args.batch_size, shuffle=False)
    if args.smoke_per_type:
        # No random draw and no change to reference selection; clearly mark subset metrics.
        counts = defaultdict(int)
        indices = []
        for index, row in enumerate(test_data.data_all):
            key = (row["cls_name"], row["anomaly"])
            if counts[key] < args.smoke_per_type:
                indices.append(index)
                counts[key] += 1
        query_data = torch.utils.data.Subset(test_data, indices)
    else:
        query_data = test_data
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    textual = TextualAdapter(model.to("cpu"), args.image_size, args.n_ctx)
    visual = VisualAdapter(args.image_size, patch_size, input_dim=input_dim, reduction=args.vl_reduction)
    pq = PQAdapter(args.image_size, patch_size, context=args.pq_context, input_dim=input_dim,
                   mid_dim=args.pq_mid_dim, layers_num=len(args.features_list))
    logger.info("Loading adapters: %s", args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path)
    for name, adapter in (("textual_learner", textual), ("visual_learner", visual), ("pq_learner", pq)):
        adapter.load_state_dict(checkpoint[name])
    for module in (model, textual, visual, pq):
        module.to(device)
        module.eval()
    textual.prepare_static_text_feature(model)
    with torch.no_grad():
        prompts, tokens = textual()
        learned_text = model.encode_text_learn(prompts, tokens).float()
    memory = build_prompt_memory(model, prompt_loader, device, test_data.obj_list,
                                 test_data.view_list, args.features_list, dpam_layer)
    logger.info("Built %s-shot references for %d classes; query samples=%d", args.k_shots, len(test_data.obj_list), len(query_data))
    return model, textual, visual, pq, learned_text, memory, query_loader, dpam_layer, device, references


def compute_branches(args, query_feats, query_patch_feats, prompt_feats, prompt_patch_feats,
                     textual, visual, pq, static_text, learned_text):
    """Expose test.py's few-shot inference steps without touching the adapters.

    Features: global [B,D], patches four [B,1+L,D], references [B,S,1+L,D].
    Returns raw [B,H,W], evaluated CPU [B,H,W], image scores CPU [B].
    The duplicated steps are limited to test.py's VA/TA/PQA and final fusion block.
    """
    import torch
    from adaptcliplib import fusion_fun
    from branch_diagnostics_utils import BRANCHES, smooth_maps

    global_vl_logit, local_vl_map = visual(query_feats, query_patch_feats, static_text)
    local_vl_map = local_vl_map[:, 1].detach()
    global_vl_score = global_vl_logit.softmax(-1)[:, 1].detach()
    global_tl_logit, local_tl_map = textual.compute_global_local_score(query_feats, query_patch_feats, learned_text)
    local_tl_map = local_tl_map[:, 1].detach()
    global_tl_score = global_tl_logit.softmax(-1)[:, 1].detach()
    global_pq_logit, local_pq_map_list, align_score_list = pq(query_feats, query_patch_feats, prompt_feats, prompt_patch_feats)
    local_pq_map_list = [value[:, 1].unsqueeze(1) for value in local_pq_map_list]
    local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
    align_score = fusion_fun(align_score_list, fusion_type="harmonic_mean")[:, 0]
    if isinstance(global_pq_logit, list):
        global_pq_score = [value.softmax(-1).unsqueeze(-1) for value in global_pq_logit]
        global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1).detach()[:, 1].detach()
    else:
        global_pq_score = global_pq_logit.softmax(-1)[:, 1].detach()
    pixel_anomaly_map = fusion_fun([local_vl_map, local_tl_map, local_pq_map], fusion_type=args.fusion_type)
    pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type="harmonic_mean")
    raw = dict(VA=local_vl_map, TA=local_tl_map, PQA=local_pq_map, ALIGN=align_score, FINAL=pixel_anomaly_map)
    # FINAL smoothing/max/fusion execute on the same devices as test.py.
    pixel_anomaly_map = smooth_maps(pixel_anomaly_map, args.sigma).to(query_feats.device)
    anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(query_feats.shape[0], -1), dim=1)
    image_anomaly_pred = fusion_fun([global_vl_score, global_tl_score, global_pq_score], fusion_type=args.fusion_type)
    image_anomaly_pred = fusion_fun([image_anomaly_pred, anomaly_map_max], fusion_type="harmonic_mean")
    pixel_anomaly_map = torch.nan_to_num(pixel_anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
    image_anomaly_pred = torch.nan_to_num(image_anomaly_pred, nan=0.0, posinf=0.0, neginf=0.0)
    evaluated = {name: smooth_maps(raw[name], args.sigma) for name in BRANCHES[:-1]}
    evaluated["FINAL"] = pixel_anomaly_map.cpu()
    scores = {"VA": global_vl_score.cpu(), "TA": global_tl_score.cpu(), "PQA": global_pq_score.cpu(),
              "ALIGN": evaluated["ALIGN"].flatten(1).max(dim=1).values, "FINAL": image_anomaly_pred.cpu()}
    return {name: value.detach().cpu() for name, value in raw.items()}, evaluated, scores


def run(args, run_dir, logger):
    """Infer each image once and flush CPU category buffers to the original evaluator."""
    import torch
    from tqdm import tqdm
    from test import prompt_association
    from tools import setup_seed
    from branch_diagnostics_utils import (
        BRANCHES, CategoryBuffer, compare_historical, require_finite, sample_key,
        save_comparison, save_tables, write_csv,
    )

    setup_seed(args.seed)
    model, textual, visual, pq, learned_text, memory, loader, dpam, device, references = initialize(args, run_dir, logger)
    pixels, images, statistics, manifest = [], [], [], []
    selected_counts = defaultdict(int)
    buffer = None
    closed_categories = set()

    def flush():
        pixel_rows, image_rows, stats_rows = buffer.finish(device, logger)
        pixels.extend(pixel_rows)
        images.extend(image_rows)
        statistics.extend(stats_rows)
        save_tables(run_dir, pixels, images, statistics)

    for items in tqdm(loader, desc="Branch diagnostics"):
        with torch.no_grad():
            query_image = items["img"].to(device)
            query_feats, query_patch_feats = model.encode_image(query_image, args.features_list, DPAM_layer=dpam)
            prompt_feats, prompt_patch_feats = prompt_association(*memory, items["cls_name"])
            raw, evaluated, scores = compute_branches(args, query_feats, query_patch_feats, prompt_feats, prompt_patch_feats,
                                                      textual, visual, pq, textual.static_text_features, learned_text)
        masks = items["img_mask"][:, 0].clone()
        masks[masks > 0.5], masks[masks <= 0.5] = 1, 0
        for name in BRANCHES:
            if raw[name].shape != masks.shape or evaluated[name].shape != masks.shape:
                raise ValueError(f"{name} map shape {raw[name].shape} != ground truth {masks.shape}")
            require_finite(f"{name} raw map", raw[name], items["img_path"])
            require_finite(f"{name} evaluated map", evaluated[name], items["img_path"])
            require_finite(f"{name} image score", scores[name], items["img_path"])
        for index, category in enumerate(items["cls_name"]):
            if buffer is None or buffer.category != category:
                if buffer is not None:
                    flush()
                    closed_categories.add(buffer.category)
                if category in closed_categories:
                    raise ValueError("Dataset category order changed; cannot flush category buffers safely.")
                buffer = CategoryBuffer(category)
            label = int(items["anomaly"][index])
            sample_type = "Anomaly" if label else "Normal"
            current_raw = {name: raw[name][index] for name in BRANCHES}
            current_maps = {name: evaluated[name][index] for name in BRANCHES}
            current_scores = {name: scores[name][index] for name in BRANCHES}
            buffer.append(masks[index], label, current_raw, current_maps, current_scores)
            path = items["img_path"][index]
            key = sample_key(path, args.test_data_path)
            selected = selected_counts[(category, label)] < args.vis_per_class
            selected_counts[(category, label)] += 1
            row = {"Category": category, "Sample Type": sample_type, "Image Path": path,
                   "Sample Key": key, "Visualization": "", "Raw Maps": "", "Evaluated Maps": ""}
            if selected and args.save_visualizations:
                output = Path("visualizations") / category / sample_type / f"{key}.png"
                save_comparison(run_dir / output, items["img"][index], masks[index].numpy(),
                                {name: value.numpy() for name, value in current_raw.items()},
                                {name: value.numpy() for name, value in current_maps.items()})
                row["Visualization"] = output.as_posix()
            if args.save_maps and (selected or args.maps_scope == "all"):
                import numpy as np
                for stage, maps, field in (("raw", current_raw, "Raw Maps"), ("evaluated", current_maps, "Evaluated Maps")):
                    output = Path("raw_maps") / category / sample_type / key / stage
                    (run_dir / output).mkdir(parents=True, exist_ok=True)
                    for name, value in maps.items():
                        np.save(run_dir / output / f"{name}.npy", value.numpy(), allow_pickle=False)
                    row[field] = output.as_posix()
            manifest.append(row)
    if buffer is None:
        raise ValueError("No test images found.")
    flush()
    write_csv(run_dir / "sample_manifest.csv", ["Category", "Sample Type", "Image Path", "Sample Key", "Visualization", "Raw Maps", "Evaluated Maps"], manifest)
    if not args.smoke_per_type:
        compare_historical(run_dir, PROJECT_ROOT, args, pixels, images, references)
    logger.info("Completed %d samples. Results: %s", len(manifest), run_dir)
    return {"status": "completed", "sample_count": len(manifest), "category_count": len({row['Category'] for row in pixels}),
            "finite_checks": "passed", "evaluation_scope": "smoke_subset" if args.smoke_per_type else "full_visa",
            "fresh_baseline_inference_compared": False}


def run_single(args):
    """Run one shot configuration; release Python/CUDA caches and log handles on exit."""
    from branch_diagnostics_utils import create_run_dir, write_json

    run_name = f"{args.dataset}_{args.k_shots}shot_seed{args.seed}" + ("_smoke" if args.smoke_per_type else "")
    run_dir = create_run_dir(PROJECT_ROOT / "results" / Path(__file__).stem, run_name)
    logger = logging.getLogger("branch_diagnostics")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(run_dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    import torch
    import scipy
    import numpy as np
    config = {**vars(args), "project_root": str(PROJECT_ROOT), "result_dir": str(run_dir),
              "metrics": [f"{level}-{name}" for level in ("I", "P") for name in ("AUROC", "AP", "F1max")],
              "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
              "metric_scale": "0-100 percent; N/A for single-label target sets", "map_max_stage": "after Gaussian smoothing",
              "statistics_stage": "before Gaussian smoothing, pooled pixels per category and sample type",
              "ap_definition": "Unmodified tools.Evaluator AUPR (trapezoidal PR area)",
              "visualization": "shared Turbo scale over smoothed maps, per image; labels include raw extrema",
              "baseline_verification": "See baseline_comparison.json; historical rounded comparison is not a fresh baseline rerun."}
    write_json(run_dir / "experiment_config.json", config)
    write_json(run_dir / "run_status.json", {"status": "running", "complete": False})
    logger.info("Results: %s; pq_context=%s", run_dir, args.pq_context)
    try:
        status = run(args, run_dir, logger)
    except KeyboardInterrupt:
        write_json(run_dir / "run_status.json", {"status": "interrupted", "complete": False,
                                                  "note": "Interrupted by user; existing outputs are partial, not full-dataset results."})
        logger.warning("Interrupted; existing CSVs are partial results: %s", run_dir)
        raise SystemExit(130)
    except Exception as error:
        write_json(run_dir / "run_status.json", {"status": "failed", "error": str(error)})
        logger.exception("Diagnostics failed; any existing CSVs are partial results.")
        raise
    else:
        write_json(run_dir / "run_status.json", status)
    finally:
        # run() has unwound before cleanup. Process exit also releases allocations
        # retained by native libraries and the CUDA context, before the next shot.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def run_sequence(argv):
    """Wait for each child to exit before launching the next; stop on failure/Ctrl+C.

    The parent imports only the standard library. Each child parses the original
    options plus one explicit shot count and owns its model, loaders and buffers.
    """
    for index, shots in enumerate((1, 2, 4), start=1):
        print(f"[Round {index}/3] Starting {shots}-shot diagnostics in a fresh process.", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), *argv, "--k_shots", str(shots)]
        with subprocess.Popen(command, cwd=str(PROJECT_ROOT)) as process:
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                # Console Ctrl+C also reaches the child, allowing it to record
                # interrupted status. Reap it before this controller exits.
                print("Interrupted; stopping the sequence and waiting for the current round to exit.", flush=True)
                try:
                    process.wait(timeout=15)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    process.terminate()
                    process.wait()
                raise SystemExit(130)
        if returncode:
            print(f"{shots}-shot failed (exit code {returncode}); later rounds were not started.", flush=True)
            raise SystemExit(returncode)
        print(f"[Round {index}/3] {shots}-shot process exited; its CPU/GPU allocations are released.", flush=True)
    print("Completed all three rounds: 1-shot, 2-shot, 4-shot.", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    for name in ("test_data_path", "checkpoint_path", "pretrained_model"):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"{name}: {getattr(args, name)}")
    if args.k_shots is None:
        run_sequence(argv)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
