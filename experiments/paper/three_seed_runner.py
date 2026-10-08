#!/usr/bin/env python3
"""Launch the paper-aligned C2Fed three-seed experiment matrix.

The runner is dry-run by default. Pass --execute only after the printed
commands and the data/checkpoint preflight have been checked. Every run gets
an isolated output directory so that runs cannot overwrite one another.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "three_seeds"
CHECKPOINT = REPO_ROOT / "checkpoints" / "imagenet21k_ViT-B_16.npz"
DATASET_CHOICES = ("cifar100", "tinyimagenet")


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    protocol: str
    local_policy: str
    run_root: Path
    command: tuple[str, ...]

    @property
    def completion_marker(self) -> Path:
        return self.run_root / "COMPLETED.json"

    @property
    def log_path(self) -> Path:
        return self.run_root / "stdout.log"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=list(DATASET_CHOICES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--protocol",
        choices=("direct",),
        default="direct",
        help="C2Fed-Direct is the primary protocol.",
    )
    parser.add_argument(
        "--local-policy",
        choices=("epoch", "k5"),
        default="epoch",
        help="Use one local epoch (primary) or five local steps (lower compute).",
    )
    parser.add_argument("--comm-rounds", type=int, default=100)
    parser.add_argument("--test-freq", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="Physical GPU ids. Jobs are assigned round-robin.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Concurrent jobs; defaults to the number of supplied GPUs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run commands. Without this flag, only validate and print them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun jobs that already have a successful completion marker.",
    )
    args = parser.parse_args(argv)
    if args.comm_rounds <= 0:
        parser.error("--comm-rounds must be positive")
    if args.test_freq <= 0:
        parser.error("--test-freq must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if not args.gpus:
        parser.error("at least one GPU id is required")
    if args.max_parallel is None:
        args.max_parallel = len(args.gpus)
    if args.max_parallel <= 0 or args.max_parallel > len(args.gpus):
        parser.error("--max-parallel must be between 1 and the number of GPUs")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def build_job(
    dataset: str,
    seed: int,
    protocol: str,
    local_policy: str,
    comm_rounds: int,
    test_freq: int,
    data_dir: Path,
    output_root: Path,
) -> Job:
    run_root = output_root / protocol / local_policy / dataset / f"seed-{seed}"
    command = (
        sys.executable,
        str(REPO_ROOT / "c2fed.py"),
        "--model_name",
        "ViT-B_16",
        "--dataset",
        dataset,
        "--partition",
        "noniid-labeluni",
        "--cls_num",
        "50",
        "--n_parties",
        "100",
        "--sample",
        "0.05",
        "--comm_round",
        str(comm_rounds),
        "--test_round",
        str(comm_rounds - 1),
        "--test_freq",
        str(test_freq),
        "--lr",
        "0.1",
        "--head_lr",
        "0.1",
        "-K",
        "0" if local_policy == "epoch" else "5",
        "--batch_size",
        "64",
        "--device",
        "cuda",
        "--train",
        "fwd",
        "--fwd_head_tuning",
        "--h",
        "0.001",
        "--clip_norm",
        "1",
        "--config_profile",
        "c2fed-direct",
        "--momentum_type",
        protocol,
        "--momentum",
        "0.9",
        "-N",
        "8",
        "--micro_batch_size",
        "8",
        "--num_micro_batch",
        "2",
        "--sample_selection",
        "uncertainty",
        "--peftmode",
        "bitfit",
        "--init_seed",
        str(seed),
        "--datadir",
        str(data_dir),
        "--logdir",
        str(run_root),
    )
    return Job(dataset, seed, protocol, local_policy, run_root, command)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        build_job(
            dataset,
            seed,
            args.protocol,
            args.local_policy,
            args.comm_rounds,
            args.test_freq,
            args.data_dir,
            args.output_root,
        )
        for dataset in args.datasets
        for seed in args.seeds
    ]


def preflight(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not CHECKPOINT.is_file():
        errors.append(f"missing ViT checkpoint: {CHECKPOINT}")
    if "tinyimagenet" in args.datasets:
        for filename in ("tinyimagenet_train.npz", "tinyimagenet_val.npz"):
            path = args.data_dir / filename
            if not path.is_file():
                errors.append(f"missing TinyImageNet file: {path}")
    return errors


def command_string(job: Job, gpu: str) -> str:
    command = " ".join(shlex.quote(part) for part in job.command)
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {command}"


def write_manifest(job: Job, gpu: str) -> None:
    job.run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": job.dataset,
        "seed": job.seed,
        "protocol": job.protocol,
        "local_policy": job.local_policy,
        "gpu": gpu,
        "command": list(job.command),
        "created_unix_time": time.time(),
    }
    (job.run_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def prepare_attempt(job: Job, gpu: str) -> None:
    """Prepare a clean attempt while retaining the previous attempt's log."""
    write_manifest(job, gpu)

    failed_marker = job.run_root / "FAILED.json"
    if failed_marker.exists():
        failed_marker.unlink()

    if job.log_path.exists():
        archived_log = job.run_root / f"stdout.previous-{time.time_ns()}.log"
        job.log_path.replace(archived_log)


def parse_gpu_memory_csv(output: str) -> dict[str, int]:
    """Parse physical GPU index and used MiB from nvidia-smi CSV output."""
    memory: dict[str, int] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            memory[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return memory


def read_gpu_memory_mib() -> dict[str, int]:
    """Return used memory for every physical GPU, or an empty map if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return parse_gpu_memory_csv(result.stdout)


def run_jobs(jobs: list[Job], args: argparse.Namespace) -> int:
    pending = [job for job in jobs if args.force or not job.completion_marker.exists()]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"Skipping {skipped} completed job(s); pass --force to rerun.")

    active: list[
        tuple[subprocess.Popen[bytes], Job, TextIO, str, float, int | None]
    ] = []
    available_gpus = list(args.gpus[: args.max_parallel])
    failures: list[tuple[Job, int]] = []

    while pending or active:
        while pending and available_gpus:
            job = pending.pop(0)
            gpu = available_gpus.pop(0)
            prepare_attempt(job, gpu)
            log_file = job.log_path.open("wb")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"Starting {job.dataset} seed={job.seed} on GPU {gpu}")
            process = subprocess.Popen(
                job.command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active.append((process, job, log_file, gpu, time.time(), None))

        time.sleep(1)
        memory_by_gpu = read_gpu_memory_mib()
        still_active: list[
            tuple[subprocess.Popen[bytes], Job, TextIO, str, float, int | None]
        ] = []
        for process, job, log_file, gpu, start_time, peak_memory_mib in active:
            current_memory_mib = memory_by_gpu.get(gpu)
            if current_memory_mib is not None:
                peak_memory_mib = max(peak_memory_mib or 0, current_memory_mib)
            return_code = process.poll()
            if return_code is None:
                still_active.append(
                    (process, job, log_file, gpu, start_time, peak_memory_mib)
                )
                continue
            log_file.close()
            elapsed = time.time() - start_time
            available_gpus.append(gpu)
            result = {
                "dataset": job.dataset,
                "seed": job.seed,
                "protocol": job.protocol,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "peak_gpu_memory_mib": peak_memory_mib,
                "finished_unix_time": time.time(),
            }
            marker = (
                job.completion_marker
                if return_code == 0
                else job.run_root / "FAILED.json"
            )
            marker.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            if return_code == 0:
                print(
                    f"Completed {job.dataset} seed={job.seed} "
                    f"in {elapsed / 3600:.2f} h"
                )
            else:
                failures.append((job, return_code))
                print(
                    f"FAILED {job.dataset} seed={job.seed} (exit {return_code}); "
                    f"see {job.log_path}",
                    file=sys.stderr,
                )
        active = still_active

    if failures:
        print(f"{len(failures)} job(s) failed.", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = build_jobs(args)
    errors = preflight(args)

    print(
        f"Matrix: protocol={args.protocol}, datasets={','.join(args.datasets)}, "
        f"seeds={','.join(map(str, args.seeds))}, runs={len(jobs)}"
    )
    for index, job in enumerate(jobs):
        gpu = args.gpus[index % args.max_parallel]
        print(command_string(job, gpu))

    if errors:
        print("\nPreflight:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if args.execute:
            return 2
    else:
        print("\nPreflight passed.")

    if not args.execute:
        print("Dry run only. Add --execute to launch the matrix.")
        return 0
    return run_jobs(jobs, args)


if __name__ == "__main__":
    raise SystemExit(main())
