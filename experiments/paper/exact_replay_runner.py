#!/usr/bin/env python3
"""Run the C2Fed Exact Replay matrix with at most two GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["cifar100", "tinyimagenet"], required=True)
    parser.add_argument("--modes", nargs="+", choices=["k5", "full_epoch"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--comm-rounds", type=int, required=True)
    parser.add_argument("--test-freq", type=int, default=5)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replay-performance-mode", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 2:
        parser.error("--max-parallel must be 1 or 2")
    if args.max_parallel > len(args.gpus):
        parser.error("--max-parallel cannot exceed the number of GPUs")
    if args.comm_rounds <= 0:
        parser.error("--comm-rounds must be positive")
    return args


def build_command(
    dataset: str,
    mode: str,
    seed: int,
    rounds: int,
    test_freq: int,
    run_root: Path,
    replay_performance_mode: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "c2fed.py"),
        "--model_name", "ViT-B_16",
        "--dataset", dataset,
        "--partition", "noniid-labeluni",
        "--cls_num", "50",
        "--n_parties", "100",
        "--sample", "0.05",
        "--comm_round", str(rounds),
        "--test_round", str(rounds - 1),
        "--test_freq", str(test_freq),
        "--lr", "0.1",
        "--head_lr", "0.1",
        "-K", "5" if mode == "k5" else "0",
        "--batch_size", "64",
        "--device", "cuda",
        "--train", "fwd",
        "--fwd_head_tuning",
        "--h", "0.001",
        "--clip_norm", "1",
        "--momentum_type", "replay-ht",
        "--momentum", "0.9",
        "-N", "8",
        "--micro_batch_size", "8",
        "--num_micro_batch", "2",
        "--sample_selection", "uncertainty",
        "--peftmode", "bitfit",
        "--init_seed", str(seed),
        "--config_profile", "c2fed-replay",
        "--datadir", "data",
        "--logdir", str(run_root),
    ]
    if replay_performance_mode:
        command.append("--replay_performance_mode")
    return command


def run_pipeline(
    gpu: str,
    jobs: list[tuple],
    asset_root: Path,
    execute: bool,
    replay_performance_mode: bool,
) -> None:
    for dataset, mode, seed, rounds, test_freq, run_root in jobs:
        run_root.mkdir(parents=True, exist_ok=True)
        command = build_command(
            dataset,
            mode,
            seed,
            rounds,
            test_freq,
            run_root,
            replay_performance_mode,
        )
        manifest = {
            "dataset": dataset,
            "mode": mode,
            "seed": seed,
            "gpu": gpu,
            "rounds": rounds,
            "replay_performance_mode": replay_performance_mode,
            "command": command,
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "created_unix_time": time.time(),
        }
        (run_root / "launch_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[QUEUE] gpu={gpu} dataset={dataset} mode={mode} seed={seed}", flush=True)
        if not execute:
            continue
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        })
        print(f"[START] gpu={gpu} dataset={dataset} mode={mode} seed={seed}", flush=True)
        with (run_root / "stdout.log").open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=asset_root,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(
            f"[DONE] gpu={gpu} dataset={dataset} mode={mode} seed={seed} rc={result.returncode}",
            flush=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"job failed; inspect {run_root / 'stdout.log'}")


def main() -> None:
    args = parse_args()
    args.asset_root = args.asset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    jobs = []
    for dataset in args.datasets:
        for mode in args.modes:
            for seed in args.seeds:
                run_root = args.output_root / mode / dataset / f"seed-{seed}"
                jobs.append((dataset, mode, seed, args.comm_rounds, args.test_freq, run_root))

    worker_count = min(args.max_parallel, len(jobs))
    pipelines = [[] for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        pipelines[index % worker_count].append(job)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_pipeline,
                args.gpus[index],
                pipelines[index],
                args.asset_root,
                args.execute,
                args.replay_performance_mode,
            )
            for index in range(worker_count)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
