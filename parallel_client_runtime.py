"""Deterministic persistent-worker runtime for client-parallel experiments.

This module contains no model or optimizer logic.  It schedules independent
client tasks, owns spawned worker processes, and returns results to a CPU
controller.  CUDA initialization belongs exclusively to worker handlers.
"""

from __future__ import annotations

import hashlib
import importlib
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch.multiprocessing as mp


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    device: str


@dataclass(frozen=True)
class RoundTask:
    round_id: int
    global_parameters: Any
    global_momentum: Any
    client_ids: tuple[int, ...]
    task_seeds: Mapping[int, int]
    expected_local_steps: Mapping[int, int]

    def validate(self) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("client_ids must be unique within a worker task")
        if set(self.client_ids) != set(self.task_seeds):
            raise ValueError("task_seeds must contain exactly the assigned clients")
        if set(self.client_ids) != set(self.expected_local_steps):
            raise ValueError(
                "expected_local_steps must contain exactly the assigned clients"
            )
        if any(self.expected_local_steps[client_id] <= 0 for client_id in self.client_ids):
            raise ValueError("expected local steps must be positive")


@dataclass(frozen=True)
class ClientResult:
    round_id: int
    client_id: int
    final_parameters: Any
    final_momentum: Any
    local_steps: int
    cuda_compute_seconds: float
    wall_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class WorkerReport:
    worker_id: int
    round_id: int
    client_results: tuple[ClientResult, ...]
    worker_wall_seconds: float
    worker_cuda_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class WorkerReady:
    worker_id: int
    device: str


@dataclass(frozen=True)
class WorkerFailure:
    worker_id: int
    round_id: int | None
    message: str
    traceback_text: str


@dataclass(frozen=True)
class StopWorker:
    pass


def derive_task_seed(base_seed: int, round_id: int, client_id: int) -> int:
    """Derive a stable task seed independent of process scheduling."""

    if min(base_seed, round_id, client_id) < 0:
        raise ValueError("base_seed, round_id, and client_id must be non-negative")
    payload = f"c2fed:{base_seed}:{round_id}:{client_id}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def deterministic_greedy_schedule(
    client_ids: Sequence[int],
    work_by_client: Mapping[int, int],
    num_workers: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign clients by longest-work-first with deterministic tie breaking."""

    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    ordered_clients = tuple(int(client_id) for client_id in client_ids)
    if len(set(ordered_clients)) != len(ordered_clients):
        raise ValueError("client_ids must be unique")
    if set(ordered_clients) != set(work_by_client):
        raise ValueError("work_by_client must contain exactly client_ids")
    if any(work_by_client[client_id] <= 0 for client_id in ordered_clients):
        raise ValueError("client work must be positive")

    original_position = {
        client_id: position for position, client_id in enumerate(ordered_clients)
    }
    scheduling_order = sorted(
        ordered_clients,
        key=lambda client_id: (-work_by_client[client_id], original_position[client_id]),
    )
    assignments: list[list[int]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    for client_id in scheduling_order:
        worker_id = min(range(num_workers), key=lambda index: (loads[index], index))
        assignments[worker_id].append(client_id)
        loads[worker_id] += work_by_client[client_id]
    return tuple(tuple(assignment) for assignment in assignments)


def order_client_results(
    reports: Sequence[WorkerReport], selected_clients: Sequence[int]
) -> tuple[ClientResult, ...]:
    """Validate worker output and restore the server's selected-client order."""

    expected = tuple(int(client_id) for client_id in selected_clients)
    if len(set(expected)) != len(expected):
        raise ValueError("selected_clients must be unique")
    by_client: dict[int, ClientResult] = {}
    round_ids: set[int] = set()
    for report in reports:
        round_ids.add(report.round_id)
        for result in report.client_results:
            if result.client_id in by_client:
                raise ValueError(f"duplicate result for client {result.client_id}")
            if result.round_id != report.round_id:
                raise ValueError("client result round does not match worker report")
            by_client[result.client_id] = result
    if len(round_ids) > 1:
        raise ValueError("worker reports belong to different rounds")
    missing = set(expected) - set(by_client)
    unexpected = set(by_client) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"client result mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return tuple(by_client[client_id] for client_id in expected)


def _resolve_symbol(path: str) -> Any:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("factory path must use 'module:attribute' syntax")
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def _worker_main(
    spec: WorkerSpec,
    request_queue: Any,
    response_queue: Any,
    handler_factory_path: str,
    handler_config: Mapping[str, Any],
) -> None:
    current_round: int | None = None
    try:
        factory = _resolve_symbol(handler_factory_path)
        handler = factory(spec.worker_id, spec.device, dict(handler_config))
        response_queue.put(WorkerReady(spec.worker_id, spec.device))
        while True:
            command = request_queue.get()
            if isinstance(command, StopWorker):
                break
            if not isinstance(command, RoundTask):
                raise TypeError(f"unexpected worker command: {type(command)!r}")
            command.validate()
            current_round = command.round_id
            report = handler.run_round(command)
            if not isinstance(report, WorkerReport):
                raise TypeError("worker handler must return WorkerReport")
            if report.worker_id != spec.worker_id:
                raise ValueError("worker handler returned the wrong worker_id")
            if report.round_id != command.round_id:
                raise ValueError("worker handler returned the wrong round_id")
            response_queue.put(report)
            current_round = None
        close = getattr(handler, "close", None)
        if close is not None:
            close()
    except BaseException as error:  # worker must propagate any fatal failure
        response_queue.put(
            WorkerFailure(
                worker_id=spec.worker_id,
                round_id=current_round,
                message=f"{type(error).__name__}: {error}",
                traceback_text=traceback.format_exc(),
            )
        )


class ParallelClientPool:
    """Own persistent spawned workers and execute one synchronized round at a time."""

    def __init__(
        self,
        worker_specs: Sequence[WorkerSpec],
        handler_factory_path: str,
        handler_config: Mapping[str, Any] | None = None,
        startup_timeout_seconds: float = 120.0,
        round_timeout_seconds: float = 600.0,
    ) -> None:
        specs = tuple(worker_specs)
        if not specs:
            raise ValueError("at least one worker is required")
        worker_ids = [spec.worker_id for spec in specs]
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker ids must be unique")
        devices = [spec.device for spec in specs]
        if len(set(devices)) != len(devices):
            raise ValueError("worker devices must be unique")
        if startup_timeout_seconds <= 0 or round_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")

        self.worker_specs = specs
        self.handler_factory_path = handler_factory_path
        self.handler_config = dict(handler_config or {})
        self.startup_timeout_seconds = startup_timeout_seconds
        self.round_timeout_seconds = round_timeout_seconds
        self._context = mp.get_context("spawn")
        self._response_queue: Any | None = None
        self._request_queues: dict[int, Any] = {}
        self._processes: dict[int, Any] = {}
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("cannot restart a closed worker pool")
        self._response_queue = self._context.Queue()
        for spec in self.worker_specs:
            request_queue = self._context.Queue(maxsize=1)
            process = self._context.Process(
                target=_worker_main,
                args=(
                    spec,
                    request_queue,
                    self._response_queue,
                    self.handler_factory_path,
                    self.handler_config,
                ),
                name=f"c2fed-worker-{spec.worker_id}",
            )
            process.start()
            self._request_queues[spec.worker_id] = request_queue
            self._processes[spec.worker_id] = process

        ready: set[int] = set()
        deadline = time.monotonic() + self.startup_timeout_seconds
        while len(ready) < len(self.worker_specs):
            message = self._next_message(deadline, "worker startup")
            if isinstance(message, WorkerFailure):
                self.close(force=True)
                raise RuntimeError(self._format_failure(message))
            if not isinstance(message, WorkerReady):
                self.close(force=True)
                raise RuntimeError(f"unexpected startup message: {message!r}")
            if message.worker_id in ready:
                self.close(force=True)
                raise RuntimeError(f"duplicate ready message from {message.worker_id}")
            ready.add(message.worker_id)
        self._started = True

    def run_round(self, tasks_by_worker: Mapping[int, RoundTask]) -> tuple[WorkerReport, ...]:
        if not self._started:
            self.start()
        expected_worker_ids = {spec.worker_id for spec in self.worker_specs}
        if set(tasks_by_worker) != expected_worker_ids:
            raise ValueError("tasks_by_worker must contain exactly all worker ids")
        round_ids = {task.round_id for task in tasks_by_worker.values()}
        if len(round_ids) != 1:
            raise ValueError("all worker tasks must belong to the same round")
        round_id = next(iter(round_ids))
        for worker_id, task in tasks_by_worker.items():
            task.validate()
            self._request_queues[worker_id].put(task)

        reports: dict[int, WorkerReport] = {}
        deadline = time.monotonic() + self.round_timeout_seconds
        while len(reports) < len(self.worker_specs):
            message = self._next_message(deadline, f"round {round_id}")
            if isinstance(message, WorkerFailure):
                self.close(force=True)
                raise RuntimeError(self._format_failure(message))
            if not isinstance(message, WorkerReport):
                self.close(force=True)
                raise RuntimeError(f"unexpected round message: {message!r}")
            if message.round_id != round_id:
                self.close(force=True)
                raise RuntimeError(
                    f"received report for round {message.round_id}, expected {round_id}"
                )
            if message.worker_id in reports:
                self.close(force=True)
                raise RuntimeError(f"duplicate report from worker {message.worker_id}")
            reports[message.worker_id] = message
        return tuple(reports[spec.worker_id] for spec in self.worker_specs)

    def _next_message(self, deadline: float, context: str) -> Any:
        assert self._response_queue is not None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self.close(force=True)
            raise TimeoutError(f"timed out during {context}")
        try:
            return self._response_queue.get(timeout=remaining)
        except queue.Empty as error:
            self.close(force=True)
            raise TimeoutError(f"timed out during {context}") from error

    @staticmethod
    def _format_failure(failure: WorkerFailure) -> str:
        return (
            f"worker {failure.worker_id} failed in round {failure.round_id}: "
            f"{failure.message}\n{failure.traceback_text}"
        )

    def close(self, force: bool = False) -> None:
        if self._closed:
            return
        if not force:
            for request_queue in self._request_queues.values():
                request_queue.put(StopWorker())
        for process in self._processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for request_queue in self._request_queues.values():
            request_queue.close()
        if self._response_queue is not None:
            self._response_queue.close()
        self._closed = True

    def __enter__(self) -> "ParallelClientPool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_object: Any) -> None:
        self.close(force=exc_type is not None)


__all__ = [
    "ClientResult",
    "ParallelClientPool",
    "RoundTask",
    "WorkerFailure",
    "WorkerReady",
    "WorkerReport",
    "WorkerSpec",
    "derive_task_seed",
    "deterministic_greedy_schedule",
    "order_client_results",
]
