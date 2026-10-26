"""Run datasets sequentially, using a fresh test.py process for every round."""

import argparse
from pathlib import Path
import subprocess
import sys


# 路径配置；常用测试参数位于文件末尾的 __main__ 中。
DATASET_ROOT = Path(r"E:\projects\datasets")
CHECKPOINT_PATH = Path("checkpoint") / "mvtec_three_adapters_ctx_seed10" / "epoch_15.pth"
SAVE_PATH = Path("results")

PROJECT_DIR = Path(__file__).resolve().parent
ADAPTER_FLAGS = (
    "--visual_learner",
    "--textual_learner",
    "--pq_learner",
    "--pq_context",
)

# 文件夹名可以保留原大小写，传给 test.py 的数据集标识使用对应名称。
DATASET_ALIASES = {
    "visa": "visa",
    "mvtec": "mvtec",
    "mvtecad": "mvtec",
    "mvtec3d": "mvtec3d",
    "mpdd": "mpdd",
    "btad": "btad",
    "realiad": "RealIAD",
    "real-iad-variety": "Real-IAD-Variety",
}


def build_command(dataset, k_shots):
    """Build one round; test.py creates the dataset/shot output directories."""
    dataset_name = DATASET_ALIASES.get(dataset.casefold(), dataset)
    return [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "test.py"),
        "--test_data_path", str(DATASET_ROOT / dataset),
        "--dataset", dataset_name,
        "--k_shots", str(k_shots),
        "--checkpoint_path", str(CHECKPOINT_PATH),
        "--save_path", str(SAVE_PATH),
        *ADAPTER_FLAGS,
    ]


def parse_args(datasets, test_mode, k_shots, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=datasets,
        help="Dataset folder names under E:\\projects\\datasets, in execution order.",
    )
    parser.add_argument(
        "--mode", choices=("zero-shot", "few-shot"), default=test_mode,
        help="Choose one mode only (default: %(default)s).",
    )
    parser.add_argument(
        "--k_shots", type=int, nargs="+",
        help="Positive shot counts for few-shot only; defaults to K_SHOTS in this file.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Validate configuration and show all commands without starting inference.",
    )
    args = parser.parse_args(argv)

    if args.mode not in ("zero-shot", "few-shot"):
        parser.error('TEST_MODE must be "zero-shot" or "few-shot".')
    if args.mode == "zero-shot":
        if args.k_shots is not None:
            parser.error("--k_shots is only allowed with --mode few-shot; zero-shot always uses 0.")
        args.shots = [0]
    else:
        args.shots = list(k_shots if args.k_shots is None else args.k_shots)
        if not args.shots or any(type(shot) is not int or shot <= 0 for shot in args.shots):
            parser.error("Few-shot requires one or more positive integer k_shots values.")
        if len(set(args.shots)) != len(args.shots):
            parser.error("k_shots values must not be repeated.")

    if not args.datasets:
        parser.error("Configure at least one dataset.")
    for dataset in args.datasets:
        if not dataset or dataset in (".", "..") or "/" in dataset or "\\" in dataset or ":" in dataset:
            parser.error(f"Use a dataset folder name, not a path: {dataset!r}")
    if not (PROJECT_DIR / "test.py").is_file():
        parser.error(f"test.py does not exist in {PROJECT_DIR}")
    if not (PROJECT_DIR / CHECKPOINT_PATH).is_file():
        parser.error(f"Checkpoint does not exist: {PROJECT_DIR / CHECKPOINT_PATH}")
    return args


def main(datasets, test_mode, k_shots, argv=None):
    args = parse_args(datasets, test_mode, k_shots, argv)
    total = len(args.datasets) * len(args.shots)
    failures = []
    completed_rounds = 0
    skipped_rounds = 0
    print(f"Mode: {args.mode}; datasets: {args.datasets}; k_shots: {args.shots}; rounds: {total}", flush=True)

    # 外层遍历数据集，内层遍历 shot；每一轮均等待独立进程退出后再继续。
    # 主进程不导入 torch、不加载模型，子进程退出时释放其 CPU/GPU 内存和 CUDA 缓存。
    for dataset_index, dataset in enumerate(args.datasets):
        for shot_index, k_shots in enumerate(args.shots):
            round_number = dataset_index * len(args.shots) + shot_index + 1
            command = build_command(dataset, k_shots)
            print(f"\n[{round_number}/{total}] {dataset}, {k_shots}-shot", flush=True)
            print(f"Output: {PROJECT_DIR / SAVE_PATH / dataset / str(k_shots)}", flush=True)
            print(subprocess.list2cmdline(command), flush=True)
            try:
                dataset_path = DATASET_ROOT / dataset
                if not dataset_path.is_dir():
                    raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")
                if not (dataset_path / "meta.json").is_file():
                    raise FileNotFoundError(f"Dataset is missing meta.json: {dataset_path}")
                if args.dry_run:
                    continue
                subprocess.run(command, cwd=PROJECT_DIR, check=True)
            except (subprocess.CalledProcessError, OSError) as exc:
                reason = f"exit code {exc.returncode}" if isinstance(exc, subprocess.CalledProcessError) else str(exc)
                skipped_shots = args.shots[shot_index + 1:]
                failures.append({
                    "dataset": dataset,
                    "k_shots": k_shots,
                    "reason": reason,
                    "skipped_shots": skipped_shots,
                })
                skipped_rounds += len(skipped_shots)
                print(
                    f"Dataset {dataset} failed at k_shots={k_shots} ({reason}). "
                    f"Skipping remaining k_shots={skipped_shots}; continuing with the next dataset.",
                    file=sys.stderr, flush=True,
                )
                break
            except KeyboardInterrupt:
                print("\nInterrupted; stopping remaining rounds.", file=sys.stderr, flush=True)
                return 130
            completed_rounds += 1
            print("Round complete; child process exited and its CPU/GPU resources were released.", flush=True)

    print("\nPreview complete." if args.dry_run else "\nAll datasets processed.", flush=True)
    if not args.dry_run:
        print(
            f"Successful rounds: {completed_rounds}; failed datasets: {len(failures)}; skipped rounds: {skipped_rounds}.",
            flush=True,
        )
    if failures:
        print("Failed dataset summary:", flush=True)
        for failure in failures:
            print(
                f"- dataset={failure['dataset']}, k_shots={failure['k_shots']}, "
                f"reason={failure['reason']}, skipped k_shots={failure['skipped_shots']}",
                flush=True,
            )
    else:
        print("No dataset failures.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    # 常用测试配置：数据集只填写 DATASET_ROOT 下的文件夹名称。
    DATASETS = ["Visa", "BTAD","MPDD","SDD","DTD","part_glass",]
    #DATASETS = ["mvtec"]
    TEST_MODE = "few-shot"  # 只能选择 "zero-shot" 或 "few-shot"
    K_SHOTS = [1, 2, 4]  # 仅用于 few-shot；可以设置一个或多个正整数

    raise SystemExit(main(DATASETS, TEST_MODE, K_SHOTS))
