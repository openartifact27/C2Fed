#!/usr/bin/env python3
"""Audit and summarize paired one-/two-worker C2Fed parallel runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class RunData:
    root: Path
    manifest: Mapping[str, Any]
    completed: Mapping[str, Any]
    measured_rows: tuple[Mapping[str, Any], ...]
    gpu_samples: Mapping[str, Any]
    final_state: Mapping[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_torch_state(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_run(root: Path) -> RunData:
    required = (
        "manifest.json",
        "COMPLETED.json",
        "round_metrics.jsonl",
        "gpu_utilization.json",
        "final_state.pt",
    )
    missing = [filename for filename in required if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete run {root}: missing {missing}")
    manifest = read_json(root / "manifest.json")
    completed = read_json(root / "COMPLETED.json")
    if completed.get("status") != "complete":
        raise ValueError(f"run is not complete: {root}")
    rows = tuple(
        json.loads(line)
        for line in (root / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    measured = tuple(row for row in rows if row.get("measured"))
    if len(measured) != int(manifest["measure_rounds"]):
        raise ValueError(
            f"measured-round mismatch in {root}: "
            f"{len(measured)} != {manifest['measure_rounds']}"
        )
    return RunData(
        root=root,
        manifest=manifest,
        completed=completed,
        measured_rows=measured,
        gpu_samples=read_json(root / "gpu_utilization.json"),
        final_state=load_torch_state(root / "final_state.pt"),
    )


def normalized_training_args(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest["training_args"])
    result.pop("logdir", None)
    result.pop("device", None)
    return result


def validate_pair(one: RunData, two: RunData, repeat_id: int) -> None:
    common_keys = (
        "dataset",
        "local_policy",
        "base_seed",
        "warmup_rounds",
        "measure_rounds",
        "repeat_id",
        "partition_fingerprint",
        "payload_ledger",
    )
    for key in common_keys:
        if one.manifest.get(key) != two.manifest.get(key):
            raise ValueError(f"paired manifest mismatch for {key} in repeat {repeat_id}")
    if one.manifest["repeat_id"] != repeat_id:
        raise ValueError(f"unexpected repeat id in {one.root}")
    if one.manifest["num_workers"] != 1 or two.manifest["num_workers"] != 2:
        raise ValueError("paired runs must be workers-1 and workers-2")
    if normalized_training_args(one.manifest) != normalized_training_args(two.manifest):
        raise ValueError(f"training configuration mismatch in repeat {repeat_id}")
    if one.manifest.get("git", {}).get("commit") != two.manifest.get("git", {}).get("commit"):
        raise ValueError(f"git commit mismatch in repeat {repeat_id}")
    selected_one = [row["selected_clients"] for row in one.measured_rows]
    selected_two = [row["selected_clients"] for row in two.measured_rows]
    if selected_one != selected_two:
        raise ValueError(f"selected-client trajectories differ in repeat {repeat_id}")


def max_state_error(one: Mapping[str, Any], two: Mapping[str, Any]) -> float:
    maximum = 0.0
    for group in ("parameters", "momentum"):
        first = one[group]
        second = two[group]
        if set(first) != set(second):
            raise ValueError(f"state keys differ for {group}")
        for key in first:
            if first[key].shape != second[key].shape or first[key].dtype != second[key].dtype:
                raise ValueError(f"state metadata differs for {group}.{key}")
            if first[key].numel():
                error = float(
                    (first[key].to(torch.float64) - second[key].to(torch.float64))
                    .abs()
                    .max()
                )
                maximum = max(maximum, error)
    return maximum


def gpu_statistics(gpu_samples: Mapping[str, Any]) -> dict[str, Any]:
    if gpu_samples.get("error"):
        return {
            "available": False,
            "error": gpu_samples["error"],
            "mean_utilization_percent": None,
            "p95_utilization_percent": None,
        }
    values = [
        float(gpu["utilization_percent"])
        for sample in gpu_samples.get("samples", [])
        for gpu in sample["gpus"]
    ]
    if not values:
        return {
            "available": False,
            "error": "no GPU utilization samples",
            "mean_utilization_percent": None,
            "p95_utilization_percent": None,
        }
    return {
        "available": True,
        "error": None,
        "mean_utilization_percent": float(np.mean(values)),
        "p95_utilization_percent": float(np.percentile(values, 95)),
    }


def summarize_run(run: RunData) -> dict[str, Any]:
    latencies = [float(row["round_latency_seconds"]) for row in run.measured_rows]
    duration = sum(latencies)
    total_steps = sum(sum(int(step) for step in row["local_steps"]) for row in run.measured_rows)
    gpu = gpu_statistics(run.gpu_samples)
    return {
        "mean_round_latency_seconds": float(statistics.mean(latencies)),
        "median_round_latency_seconds": float(statistics.median(latencies)),
        "p95_round_latency_seconds": float(np.percentile(latencies, 95)),
        "mean_worker_span_seconds": float(
            statistics.mean(float(row["worker_span_seconds"]) for row in run.measured_rows)
        ),
        "mean_aggregation_seconds": float(
            statistics.mean(float(row["aggregation_seconds"]) for row in run.measured_rows)
        ),
        "clients_per_second": 5.0 * len(run.measured_rows) / duration,
        "local_steps_per_second": total_steps / duration,
        "peak_memory_bytes": max(
            int(value)
            for row in run.measured_rows
            for value in row["worker_peak_memory_bytes"]
        ),
        **gpu,
    }


def sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def grade_speedup(speedup: float, correct: bool, evidence_complete: bool) -> str:
    if not correct or speedup < 1.15:
        return "red"
    if speedup >= 1.40 and evidence_complete:
        return "green"
    return "yellow"


def summarize_dataset(
    root: Path,
    dataset: str,
    local_policy: str,
    repeats: Sequence[int],
    tolerance: float,
    correctness_audit_root: Path | None = None,
) -> dict[str, Any]:
    repeat_rows = []
    for repeat_id in repeats:
        one = load_run(
            root / dataset / local_policy / "workers-1" / f"repeat-{repeat_id}"
        )
        two = load_run(
            root / dataset / local_policy / "workers-2" / f"repeat-{repeat_id}"
        )
        validate_pair(one, two, repeat_id)
        one_summary = summarize_run(one)
        two_summary = summarize_run(two)
        error = max_state_error(one.final_state, two.final_state)
        speedup = (
            one_summary["mean_round_latency_seconds"]
            / two_summary["mean_round_latency_seconds"]
        )
        repeat_rows.append(
            {
                "repeat_id": repeat_id,
                "one_worker": one_summary,
                "two_worker": two_summary,
                "max_state_error": error,
                "state_equivalent": error <= tolerance,
                "speedup": speedup,
                "parallel_efficiency": speedup / 2.0,
                "schedule_ceiling_efficiency": speedup / (5.0 / 3.0),
            }
        )

    speedups = [row["speedup"] for row in repeat_rows]
    timing_errors = [row["max_state_error"] for row in repeat_rows]
    if correctness_audit_root is None:
        correctness_error = max(timing_errors)
        correctness_source = "timing_runs"
    else:
        audit_one = load_run(
            correctness_audit_root
            / dataset
            / local_policy
            / "workers-1"
            / "repeat-0"
        )
        audit_two = load_run(
            correctness_audit_root
            / dataset
            / local_policy
            / "workers-2"
            / "repeat-0"
        )
        validate_pair(audit_one, audit_two, 0)
        if not (
            normalized_training_args(audit_one.manifest).get("deterministic")
            and normalized_training_args(audit_two.manifest).get("deterministic")
        ):
            raise ValueError(
                f"correctness audit for {dataset} must use --deterministic"
            )
        correctness_error = max_state_error(
            audit_one.final_state, audit_two.final_state
        )
        correctness_source = str(correctness_audit_root)
    correct = correctness_error <= tolerance
    evidence_complete = all(
        row[worker]["available"]
        for row in repeat_rows
        for worker in ("one_worker", "two_worker")
    )
    mean_speedup = float(statistics.mean(speedups))
    return {
        "dataset": dataset,
        "local_policy": local_policy,
        "repeats": repeat_rows,
        "mean_speedup": mean_speedup,
        "speedup_sample_std": sample_std(speedups),
        "mean_parallel_efficiency": mean_speedup / 2.0,
        "mean_schedule_ceiling_efficiency": mean_speedup / (5.0 / 3.0),
        "max_state_error": correctness_error,
        "timing_max_state_error": max(timing_errors),
        "correctness_source": correctness_source,
        "correctness_tolerance": tolerance,
        "correct": correct,
        "evidence_complete": evidence_complete,
        "grade": grade_speedup(mean_speedup, correct, evidence_complete),
    }


def overall_grade(dataset_summaries: Sequence[Mapping[str, Any]]) -> str:
    grades = {summary["grade"] for summary in dataset_summaries}
    if grades == {"green"}:
        return "green"
    if "red" in grades:
        return "red"
    return "yellow"


def write_outputs(root: Path, summary: Mapping[str, Any]) -> None:
    (root / "parallel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (root / "parallel_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "dataset",
            "local_policy",
            "grade",
            "mean_speedup",
            "speedup_sample_std",
            "mean_parallel_efficiency",
            "mean_schedule_ceiling_efficiency",
            "max_state_error",
            "evidence_complete",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_summary in summary["datasets"]:
            writer.writerow({key: dataset_summary[key] for key in fieldnames})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="Write summaries here without modifying the input artifacts.")
    parser.add_argument("--datasets", nargs="+", choices=("cifar100", "tinyimagenet"), default=["cifar100", "tinyimagenet"])
    parser.add_argument("--local-policy", choices=("epoch", "k5"), default="epoch")
    parser.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--correctness-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--correctness-audit-root",
        type=Path,
        help=(
            "Root containing paired deterministic repeat-0 runs. When set, "
            "correctness is gated there and state drift in native timing runs "
            "is reported separately."
        ),
    )
    parser.add_argument(
        "--direct-reference-status",
        choices=("pending", "passed"),
        default="pending",
        help="Status of the separately audited Direct sequential-reference gate.",
    )
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.expanduser().resolve()
    if args.correctness_audit_root is not None:
        args.correctness_audit_root = args.correctness_audit_root.expanduser().resolve()
    if args.correctness_tolerance < 0:
        parser.error("--correctness-tolerance must be non-negative")
    if len(set(args.repeats)) != len(args.repeats):
        parser.error("--repeats must be unique")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = [
        summarize_dataset(
            args.root,
            dataset,
            args.local_policy,
            args.repeats,
            args.correctness_tolerance,
            args.correctness_audit_root,
        )
        for dataset in args.datasets
    ]
    summary = {
        "datasets": datasets,
        "overall_grade": overall_grade(datasets),
        "merge_status": "pending_user_confirmation",
        "direct_reference_gate": args.direct_reference_status,
    }
    output_root = args.output_root if args.output_root is not None else args.root
    output_root.mkdir(parents=True, exist_ok=True)
    write_outputs(output_root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
