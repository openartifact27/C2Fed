#!/usr/bin/env python3
"""Run the isolated one-/two-GPU C2Fed client-parallel benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federated_protocol import direct_payload_bytes
from local_training import train_local_fwd
from parallel_client_runtime import (
    ClientResult,
    ParallelClientPool,
    RoundTask,
    WorkerReport,
    WorkerSpec,
    derive_task_seed,
    deterministic_greedy_schedule,
    order_client_results,
)
from utils import partition_data


DATASETS = ("cifar100", "tinyimagenet")
PREWARM_ROUND_ID = (1 << 31) - 1


def git_revision() -> dict[str, Any]:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "dirty": bool(run_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def query_gpu_status() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": float(fields[2]),
                "memory_used_mib": float(fields[3]),
                "utilization_percent": float(fields[4]),
            }
        )
    return rows


class NvidiaSmiSampler:
    def __init__(self, gpu_ids: Sequence[str], interval_seconds: float):
        if interval_seconds <= 0:
            raise ValueError("GPU sample interval must be positive")
        self.gpu_ids = {int(gpu_id) for gpu_id in gpu_ids}
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rows = [
                    row for row in query_gpu_status() if row["index"] in self.gpu_ids
                ]
                if {row["index"] for row in rows} != self.gpu_ids:
                    raise RuntimeError("nvidia-smi did not return every requested GPU")
                self.samples.append({"monotonic_time": time.monotonic(), "gpus": rows})
            except BaseException as error:
                self.error = f"{type(error).__name__}: {error}"
                return
            self._stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2 * self.interval_seconds))

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_ids": sorted(self.gpu_ids),
            "interval_seconds": self.interval_seconds,
            "error": self.error,
            "samples": self.samples,
        }


def seed_everything(seed: int, include_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_num_classes(dataset: str) -> int:
    if dataset == "cifar100":
        return 100
    if dataset == "tinyimagenet":
        return 200
    raise ValueError(f"unsupported benchmark dataset: {dataset}")


def build_training_args(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    base_seed: int,
    local_policy: str,
    deterministic: bool = False,
) -> SimpleNamespace:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if local_policy not in ("epoch", "k5"):
        raise ValueError("local_policy must be 'epoch' or 'k5'")
    return SimpleNamespace(
        model_name="ViT-B_16",
        dataset=dataset,
        partition="noniid-labeluni",
        cls_num=50,
        batch_size=64,
        lr=0.1,
        head_lr=0.1,
        epochs=1,
        n_parties=100,
        init_seed=base_seed,
        datadir=str(data_dir),
        logdir=str(output_dir),
        beta=0.5,
        optimizer="sgd",
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0,
        h=0.001,
        perturb_num=8,
        batch_num=0 if local_policy == "epoch" else 5,
        micro_batch_size=8,
        num_micro_batch=2,
        mask_ratio=0.0,
        train="fwd",
        peftmode="bitfit",
        adapter_dim=8,
        alpha=8,
        r=8,
        fwd_head_tuning=True,
        partial_layer_training=False,
        sample_selection="uncertainty",
        momentum_type="direct",
        clip_norm=1.0,
        deterministic=deterministic,
        device="cpu",
    )


def build_vit_model(args: SimpleNamespace, checkpoint: Path, device: str, compile_model: bool):
    from models.vit_models_fwd import CONFIGS, VisionTransformerForward

    config = CONFIGS["ViT-B_16"]
    model = VisionTransformerForward(
        config,
        img_size=224,
        num_classes=dataset_num_classes(args.dataset),
        vis=True,
        args=args,
    )
    model.load_from(np.load(checkpoint))
    model.freeze()
    model.to(device)
    if compile_model:
        model = torch.compile(model, backend="inductor")
    return model


def partition_fingerprint(index_map: Mapping[int, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for client_id in sorted(index_map):
        indices = np.asarray(index_map[client_id], dtype=np.int64)
        digest.update(int(client_id).to_bytes(8, "little", signed=False))
        digest.update(len(indices).to_bytes(8, "little", signed=False))
        digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def local_steps_from_sizes(
    index_map: Mapping[int, np.ndarray], batch_size: int, batch_num: int
) -> dict[int, int]:
    result: dict[int, int] = {}
    for client_id, indices in index_map.items():
        sample_count = len(indices)
        if sample_count <= 0:
            raise ValueError(f"client {client_id} has no training samples")
        effective_batch = min(batch_size, sample_count)
        epoch_steps = sample_count // effective_batch
        result[int(client_id)] = batch_num if batch_num > 0 else epoch_steps
    return result


def clone_cpu_state(state: Mapping[Any, torch.Tensor]) -> OrderedDict:
    return OrderedDict(
        (key, value.detach().to("cpu").contiguous().clone())
        for key, value in state.items()
    )


def aggregate_client_results(
    ordered_results: Sequence[ClientResult],
) -> tuple[OrderedDict[str, torch.Tensor], dict[int, torch.Tensor]]:
    if not ordered_results:
        raise ValueError("cannot aggregate an empty client result list")
    parameter_keys = tuple(ordered_results[0].final_parameters)
    momentum_keys = tuple(sorted(ordered_results[0].final_momentum))
    for result in ordered_results:
        if tuple(result.final_parameters) != parameter_keys:
            raise ValueError(f"parameter key mismatch for client {result.client_id}")
        if tuple(sorted(result.final_momentum)) != momentum_keys:
            raise ValueError(f"momentum key mismatch for client {result.client_id}")

    parameters = OrderedDict()
    for name in parameter_keys:
        values = [result.final_parameters[name] for result in ordered_results]
        reference = values[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype for value in values):
            raise ValueError(f"parameter metadata mismatch for {name}")
        parameters[name] = torch.stack(values, dim=0).mean(dim=0)

    momentum: dict[int, torch.Tensor] = {}
    for index in momentum_keys:
        values = [result.final_momentum[index] for result in ordered_results]
        reference = values[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype for value in values):
            raise ValueError(f"momentum metadata mismatch for index {index}")
        momentum[index] = torch.stack(values, dim=0).mean(dim=0)
    return parameters, momentum


def state_fingerprint(
    parameters: Mapping[str, torch.Tensor], momentum: Mapping[int, torch.Tensor]
) -> str:
    digest = hashlib.sha256()
    for name, value in parameters.items():
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    for index in sorted(momentum):
        tensor = momentum[index].detach().to("cpu").contiguous()
        digest.update(int(index).to_bytes(8, "little", signed=False))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class C2FedClientHandler:
    def __init__(self, worker_id: int, device: str, config: Mapping[str, Any]):
        deterministic = bool(config["training_args"].get("deterministic", False))
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        # Match the main C2Fed execution mode.
        torch.set_float32_matmul_precision("high")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the client-parallel benchmark")
        self.worker_id = worker_id
        self.device = device
        torch.cuda.set_device(torch.device(device))
        args_dict = dict(config["training_args"])
        args_dict["device"] = device
        self.args = SimpleNamespace(**args_dict)
        self.base_seed = int(config["base_seed"])
        self.expected_partition_fingerprint = str(config["partition_fingerprint"])
        checkpoint = Path(config["checkpoint"])

        seed_everything(self.base_seed, include_cuda=True)
        train_ds, _, train_indices, _, _, _ = partition_data(
            self.args.dataset,
            self.args.datadir,
            self.args.partition,
            self.args.n_parties,
            beta=self.args.beta,
            logdir=self.args.logdir,
            args=self.args,
        )
        actual_fingerprint = partition_fingerprint(train_indices)
        if actual_fingerprint != self.expected_partition_fingerprint:
            raise RuntimeError(
                "worker partition does not match controller: "
                f"{actual_fingerprint} != {self.expected_partition_fingerprint}"
            )

        self.loaders: dict[int, tuple[DataLoader, torch.Generator]] = {}
        for client_id, indices in train_indices.items():
            subset = Subset(train_ds, indices)
            generator = torch.Generator()
            loader = DataLoader(
                dataset=subset,
                batch_size=min(self.args.batch_size, len(subset)),
                shuffle=True,
                drop_last=True,
                generator=generator,
            )
            self.loaders[int(client_id)] = (loader, generator)

        self.model = build_vit_model(
            self.args, checkpoint, device=self.device, compile_model=True
        )
        self.trainable_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

    def _optimizer_state(self, momentum: Mapping[int, torch.Tensor]) -> dict[str, Any]:
        return {
            "state": {
                int(index): {
                    "momentum_buffer": value.detach().to(self.device).clone()
                }
                for index, value in momentum.items()
            }
        }

    def _run_client(self, task: RoundTask, client_id: int) -> ClientResult:
        task_seed = int(task.task_seeds[client_id])
        seed_everything(task_seed, include_cuda=True)
        loader, loader_generator = self.loaders[client_id]
        loader_generator.manual_seed(task_seed)
        expected_steps = self.args.batch_num if self.args.batch_num > 0 else len(loader)
        if expected_steps != task.expected_local_steps[client_id]:
            raise RuntimeError(
                f"client {client_id} local-step mismatch: "
                f"worker={expected_steps}, controller={task.expected_local_steps[client_id]}"
            )

        self.model.load_state_dict(task.global_parameters, strict=False)
        parameter_dict = {
            "round": task.round_id,
            "trainable_layer_names": self.trainable_names,
            "train_dataloader": loader,
            "optimizer_state": self._optimizer_state(task.global_momentum),
        }

        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.model, returned_optimizer = train_local_fwd(
            self.model, self.args, parameter_dict, None
        )
        end_event.record()
        torch.cuda.synchronize(self.device)
        wall_seconds = time.perf_counter() - wall_start
        cuda_seconds = start_event.elapsed_time(end_event) / 1000.0

        final_parameters = OrderedDict(
            (name, parameter.detach().to("cpu").contiguous().clone())
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        final_momentum = {
            int(index): state["momentum_buffer"].detach().to("cpu").contiguous().clone()
            for index, state in returned_optimizer["state"].items()
        }
        return ClientResult(
            round_id=task.round_id,
            client_id=client_id,
            final_parameters=final_parameters,
            final_momentum=final_momentum,
            local_steps=expected_steps,
            cuda_compute_seconds=cuda_seconds,
            wall_seconds=wall_seconds,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )

    def run_round(self, task: RoundTask) -> WorkerReport:
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        results = tuple(self._run_client(task, client_id) for client_id in task.client_ids)
        torch.cuda.synchronize(self.device)
        wall_seconds = time.perf_counter() - start
        return WorkerReport(
            worker_id=self.worker_id,
            round_id=task.round_id,
            client_results=results,
            worker_wall_seconds=wall_seconds,
            worker_cuda_seconds=sum(result.cuda_compute_seconds for result in results),
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def build_c2fed_handler(worker_id: int, device: str, config: Mapping[str, Any]):
    return C2FedClientHandler(worker_id, device, config)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--num-workers", type=int, choices=(1, 2), required=True)
    parser.add_argument("--local-policy", choices=("epoch", "k5"), default="epoch")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--measure-rounds", type=int, default=20)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--round-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic CUDA kernels for correctness audits. "
            "Leave disabled for protocol-aligned wall-clock measurements."
        ),
    )
    parser.add_argument(
        "--allow-busy-gpus",
        action="store_true",
        help="Allow launch when preflight observes >10%% utilization or >1 GiB used.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the benchmark. Without this flag, validate and print the manifest.",
    )
    args = parser.parse_args(argv)
    if args.num_workers != len(args.gpus):
        parser.error("--num-workers must equal the number of --gpus")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must be unique")
    if min(args.base_seed, args.warmup_rounds, args.measure_rounds, args.repeat_id) < 0:
        parser.error("seed, rounds, and repeat id must be non-negative")
    if args.measure_rounds <= 0:
        parser.error("--measure-rounds must be positive")
    if args.gpu_sample_interval <= 0:
        parser.error("--gpu-sample-interval must be positive")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def preflight(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not args.checkpoint.is_file():
        errors.append(f"missing checkpoint: {args.checkpoint}")
    if not args.data_dir.is_dir():
        errors.append(f"missing data directory: {args.data_dir}")
    if args.dataset == "tinyimagenet":
        for filename in ("tinyimagenet_train.npz", "tinyimagenet_val.npz"):
            if not (args.data_dir / filename).is_file():
                errors.append(f"missing TinyImageNet file: {args.data_dir / filename}")
    if args.execute:
        try:
            status = query_gpu_status()
            by_index = {row["index"]: row for row in status}
            missing = [gpu for gpu in args.gpus if int(gpu) not in by_index]
            if missing:
                errors.append(f"nvidia-smi cannot see requested GPUs: {missing}")
            selected = [by_index[int(gpu)] for gpu in args.gpus if int(gpu) in by_index]
            if len({(row["name"], row["memory_total_mib"]) for row in selected}) > 1:
                errors.append("requested GPUs do not have identical model and memory")
            if not args.allow_busy_gpus:
                busy = [
                    row
                    for row in selected
                    if row["utilization_percent"] > 10 or row["memory_used_mib"] > 1024
                ]
                if busy:
                    errors.append(f"requested GPUs are not idle: {busy}")
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            errors.append(f"GPU preflight failed: {type(error).__name__}: {error}")
    return errors


def run_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_root
        / args.dataset
        / args.local_policy
        / f"workers-{args.num_workers}"
        / f"repeat-{args.repeat_id}"
    )


def prepare_controller(
    args: argparse.Namespace, training_args: SimpleNamespace
) -> tuple[OrderedDict[str, torch.Tensor], dict[int, int], str]:
    seed_everything(args.base_seed, include_cuda=False)
    _, _, train_indices, _, _, _ = partition_data(
        training_args.dataset,
        training_args.datadir,
        training_args.partition,
        training_args.n_parties,
        beta=training_args.beta,
        logdir=training_args.logdir,
        args=training_args,
    )
    fingerprint = partition_fingerprint(train_indices)
    local_steps = local_steps_from_sizes(
        train_indices, training_args.batch_size, training_args.batch_num
    )
    seed_everything(args.base_seed, include_cuda=False)
    model = build_vit_model(
        training_args,
        args.checkpoint,
        device="cpu",
        compile_model=True,
    )
    parameters = clone_cpu_state(
        OrderedDict(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    del model
    return parameters, local_steps, fingerprint


def build_round_tasks(
    round_id: int,
    selected_clients: Sequence[int],
    schedule: Sequence[Sequence[int]],
    parameters: Mapping[str, torch.Tensor],
    momentum: Mapping[int, torch.Tensor],
    local_steps: Mapping[int, int],
    base_seed: int,
) -> dict[int, RoundTask]:
    selected_set = set(selected_clients)
    scheduled = [client_id for group in schedule for client_id in group]
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != selected_set:
        raise ValueError("schedule must contain every selected client exactly once")
    tasks: dict[int, RoundTask] = {}
    for worker_id, clients in enumerate(schedule):
        clients_tuple = tuple(int(client_id) for client_id in clients)
        tasks[worker_id] = RoundTask(
            round_id=round_id,
            global_parameters=parameters,
            global_momentum=momentum,
            client_ids=clients_tuple,
            task_seeds={
                client_id: derive_task_seed(base_seed, round_id, client_id)
                for client_id in clients_tuple
            },
            expected_local_steps={
                client_id: int(local_steps[client_id]) for client_id in clients_tuple
            },
        )
    return tasks


def build_prewarm_tasks(
    num_workers: int,
    parameters: Mapping[str, torch.Tensor],
    local_steps: Mapping[int, int],
    base_seed: int,
    client_id: int = 0,
) -> dict[int, RoundTask]:
    """Build discarded tasks that materialize lazy compiled kernels per worker.

    Compilation must finish before the first federated client seed is installed;
    otherwise one- and two-worker executions can consume different random streams
    while compiling the first client update. Every worker receives the same
    baseline state and the outputs are deliberately ignored.
    """

    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if client_id not in local_steps:
        raise ValueError(f"missing local-step count for prewarm client {client_id}")
    task_seed = derive_task_seed(base_seed, PREWARM_ROUND_ID, client_id)
    return {
        worker_id: RoundTask(
            round_id=PREWARM_ROUND_ID,
            global_parameters=parameters,
            global_momentum={},
            client_ids=(client_id,),
            task_seeds={client_id: task_seed},
            expected_local_steps={client_id: int(local_steps[client_id])},
        )
        for worker_id in range(num_workers)
    }


def manifest_dict(
    args: argparse.Namespace,
    training_args: SimpleNamespace,
    partition_hash: str | None,
    payload_ledger: Mapping[str, int] | None,
) -> dict[str, Any]:
    gpu_inventory = None
    if args.execute:
        gpu_inventory = [
            row for row in query_gpu_status() if str(row["index"]) in set(args.gpus)
        ]
    return {
        "dataset": args.dataset,
        "local_policy": args.local_policy,
        "base_seed": args.base_seed,
        "warmup_rounds": args.warmup_rounds,
        "measure_rounds": args.measure_rounds,
        "repeat_id": args.repeat_id,
        "num_workers": args.num_workers,
        "gpus": list(args.gpus),
        "data_dir": str(args.data_dir),
        "checkpoint": str(args.checkpoint),
        "training_args": vars(training_args),
        "partition_fingerprint": partition_hash,
        "payload_ledger": dict(payload_ledger) if payload_ledger is not None else None,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "gpu_inventory": gpu_inventory,
        "git": git_revision(),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> Path:
    run_root = run_directory(args)
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    training_args = build_training_args(
        args.dataset,
        args.data_dir,
        run_root,
        args.base_seed,
        args.local_policy,
        deterministic=args.deterministic,
    )
    manifest_path = run_root / "manifest.json"
    write_json(manifest_path, manifest_dict(args, training_args, None, None))

    try:
        parameters, local_steps, partition_hash = prepare_controller(args, training_args)
        payload_ledger = direct_payload_bytes(parameters)
        write_json(
            manifest_path,
            manifest_dict(args, training_args, partition_hash, payload_ledger),
        )
        handler_config = {
            "training_args": vars(training_args),
            "base_seed": args.base_seed,
            "partition_fingerprint": partition_hash,
            "checkpoint": str(args.checkpoint),
        }
        specs = tuple(
            WorkerSpec(worker_id=index, device=f"cuda:{gpu}")
            for index, gpu in enumerate(args.gpus)
        )
        momentum: dict[int, torch.Tensor] = {}
        selection_rng = np.random.default_rng(args.base_seed)
        clients = np.arange(training_args.n_parties)
        measured_rows: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        total_rounds = args.warmup_rounds + args.measure_rounds
        gpu_sampler = NvidiaSmiSampler(args.gpus, args.gpu_sample_interval)

        try:
            with ParallelClientPool(
                specs,
                "experiments.paper.client_parallel_runner:build_c2fed_handler",
                handler_config,
                startup_timeout_seconds=args.startup_timeout,
                round_timeout_seconds=args.round_timeout,
            ) as pool:
                prewarm_start = time.perf_counter()
                pool.run_round(
                    build_prewarm_tasks(
                        args.num_workers,
                        parameters,
                        local_steps,
                        args.base_seed,
                    )
                )
                prewarm_seconds = time.perf_counter() - prewarm_start
                for round_id in range(total_rounds):
                    if round_id == args.warmup_rounds:
                        gpu_sampler.start()
                    selected = tuple(
                        int(value)
                        for value in selection_rng.choice(clients, size=5, replace=False)
                    )
                    work = {client_id: local_steps[client_id] for client_id in selected}
                    schedule = deterministic_greedy_schedule(
                        selected, work, args.num_workers
                    )
                    tasks = build_round_tasks(
                        round_id,
                        selected,
                        schedule,
                        parameters,
                        momentum,
                        local_steps,
                        args.base_seed,
                    )
                    round_start = time.perf_counter()
                    reports = pool.run_round(tasks)
                    collection_seconds = time.perf_counter() - round_start
                    ordered = order_client_results(reports, selected)
                    aggregation_start = time.perf_counter()
                    parameters, momentum = aggregate_client_results(ordered)
                    aggregation_seconds = time.perf_counter() - aggregation_start
                    row = {
                        "round": round_id,
                        "measured": round_id >= args.warmup_rounds,
                        "selected_clients": list(selected),
                        "schedule": [list(group) for group in schedule],
                        "local_steps": [result.local_steps for result in ordered],
                        "round_latency_seconds": (
                            collection_seconds + aggregation_seconds
                        ),
                        "pool_seconds": collection_seconds,
                        "aggregation_seconds": aggregation_seconds,
                        "worker_span_seconds": max(
                            report.worker_wall_seconds for report in reports
                        ),
                        "worker_cuda_seconds": [
                            report.worker_cuda_seconds for report in reports
                        ],
                        "worker_wall_seconds": [
                            report.worker_wall_seconds for report in reports
                        ],
                        "worker_peak_memory_bytes": [
                            report.peak_memory_bytes for report in reports
                        ],
                        "state_fingerprint": state_fingerprint(parameters, momentum),
                    }
                    all_rows.append(row)
                    if row["measured"]:
                        measured_rows.append(row)
        finally:
            gpu_sampler.stop()
        write_json(run_root / "gpu_utilization.json", gpu_sampler.as_dict())

        torch.save(
            {"parameters": parameters, "momentum": momentum},
            run_root / "final_state.pt",
        )
        with (run_root / "round_metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary = {
            "status": "complete",
            "worker_prewarm_seconds": prewarm_seconds,
            "measured_rounds": len(measured_rows),
            "mean_round_latency_seconds": float(
                np.mean([row["round_latency_seconds"] for row in measured_rows])
            ),
            "median_round_latency_seconds": float(
                np.median([row["round_latency_seconds"] for row in measured_rows])
            ),
            "p95_round_latency_seconds": float(
                np.percentile(
                    [row["round_latency_seconds"] for row in measured_rows], 95
                )
            ),
            "mean_worker_span_seconds": float(
                np.mean([row["worker_span_seconds"] for row in measured_rows])
            ),
            "mean_aggregation_seconds": float(
                np.mean([row["aggregation_seconds"] for row in measured_rows])
            ),
            "final_state_fingerprint": state_fingerprint(parameters, momentum),
            "gpu_sampler_error": gpu_sampler.error,
            "gpu_sample_count": len(gpu_sampler.samples),
        }
        write_json(run_root / "COMPLETED.json", summary)
        return run_root
    except BaseException as error:
        write_json(
            run_root / "FAILED.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = preflight(args)
    training_args = build_training_args(
        args.dataset,
        args.data_dir,
        run_directory(args),
        args.base_seed,
        args.local_policy,
        deterministic=args.deterministic,
    )
    print(json.dumps(manifest_dict(args, training_args, None, None), indent=2, sort_keys=True))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if not args.execute:
        print("Dry run only; pass --execute after checking the manifest.")
        return 0
    run_root = execute(args)
    print(f"Completed: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
