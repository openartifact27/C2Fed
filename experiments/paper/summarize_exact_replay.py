#!/usr/bin/env python3
"""Audit and summarize the exact Head-Tuning/PEFT replay experiment matrix.

The script treats the experiment artifacts as the source of truth.  It checks
all per-client replay traces, computes exact logical communication from the
recorded ledgers, and compares the completed three-seed matrix with matched
C2Fed-Direct (4S) runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("cifar100", "tinyimagenet")
MODES = ("k5", "full_epoch")
SEEDS = (0, 1, 2)
TARGETS = {"cifar100": 0.87, "tinyimagenet": 0.83}
EXPECTED_ROUNDS = 100
CLIENTS_PER_ROUND = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=REPO_ROOT / "artifacts/exact_replay",
    )
    parser.add_argument(
        "--direct-root",
        type=Path,
        default=REPO_ROOT / "artifacts/three_seeds",
        help="Root produced by three_seed_runner.py for matched Direct runs.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit a partial audit instead of failing on unfinished runs.",
    )
    args = parser.parse_args(argv)
    args.replay_root = args.replay_root.expanduser().resolve()
    args.direct_root = args.direct_root.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    return args


def one_file(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no {pattern!r} below {root}")
    return candidates[-1]


def read_result_rows(seed_root: Path) -> tuple[Path, list[dict[str, float]]]:
    path = one_file(seed_root, "**/results-*.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, float]] = []
    required = {
        "round",
        "time",
        "test_avg_loss",
        "test_avg_acc",
        "local_mean_acc",
        "local_min_acc",
    }
    for raw in raw_rows:
        if not required.issubset(raw):
            raise ValueError(f"missing CSV columns in {path}: {required - set(raw)}")
        row = {key: float(raw[key]) for key in required}
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"non-finite result in {path}: {raw}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty results CSV: {path}")
    return path, rows


def direct_seed_root(direct_root: Path, dataset: str, mode: str, seed: int) -> Path:
    local_policy = "k5" if mode == "k5" else "epoch"
    return direct_root / "direct" / local_policy / dataset / f"seed-{seed}"


def curve_summary(rows: list[dict[str, float]], target: float) -> dict[str, Any]:
    accuracies = [row["test_avg_acc"] for row in rows]
    best_index = max(range(len(rows)), key=lambda index: accuracies[index])
    target_row = next((row for row in rows if row["test_avg_acc"] >= target), None)
    return {
        "last_round": int(rows[-1]["round"]),
        "measurements": len(rows),
        "final_accuracy": accuracies[-1],
        "final_loss": rows[-1]["test_avg_loss"],
        "best_accuracy": accuracies[best_index],
        "best_round": int(rows[best_index]["round"]),
        "rounds_to_target": int(target_row["round"]) if target_row else None,
        "mean_measured_client_time_seconds": statistics.fmean(
            row["time"] for row in rows
        ),
    }


def wallclock_seconds(seed_root: Path, complete: bool) -> float | None:
    """Approximate elapsed process time from manifest creation to last log write."""
    if not complete:
        return None
    manifest_path = seed_root / "launch_manifest.json"
    log_path = seed_root / "stdout.log"
    if not manifest_path.is_file() or not log_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = float(manifest["created_unix_time"])
    return max(0.0, log_path.stat().st_mtime - start)


def read_replay_records(seed_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    path = one_file(seed_root, "**/replay_diagnostics.jsonl")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return path, records


def exact_seed_record(
    replay_root: Path, dataset: str, mode: str, seed: int
) -> dict[str, Any]:
    seed_root = replay_root / mode / dataset / f"seed-{seed}"
    result_path, rows = read_result_rows(seed_root)
    diagnostic_path, diagnostics = read_replay_records(seed_root)
    expected_traces = EXPECTED_ROUNDS * CLIENTS_PER_ROUND
    curve = curve_summary(rows, TARGETS[dataset])
    complete = (
        curve["last_round"] == EXPECTED_ROUNDS - 1
        and len(diagnostics) == expected_traces
    )

    parameter_errors = [float(record["parameter_max_abs_error"]) for record in diagnostics]
    momentum_errors = [float(record["momentum_max_abs_error"]) for record in diagnostics]
    payloads = [record["payload"] for record in diagnostics]
    local_steps = [int(record["local_steps"]) for record in diagnostics]
    rounds = [int(record["round"]) for record in diagnostics]
    if diagnostics and any(
        int(record["jvp_count"])
        != int(record["local_steps"]) * int(record["payload"]["jvps_per_step"])
        for record in diagnostics
    ):
        raise ValueError(f"JVP-count mismatch in {diagnostic_path}")

    total_payload = sum(int(payload["bidirectional"]) for payload in payloads)
    total_four_s = sum(int(payload["four_s_reference"]) for payload in payloads)
    return {
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        "complete": complete,
        **curve,
        "diagnostic_traces": len(diagnostics),
        "expected_diagnostic_traces": expected_traces,
        "completed_replay_rounds": len(set(rounds)),
        "parameter_max_abs_error": max(parameter_errors, default=None),
        "momentum_max_abs_error": max(momentum_errors, default=None),
        "local_steps_min": min(local_steps, default=None),
        "local_steps_max": max(local_steps, default=None),
        "payload_per_client_min_bytes": min(
            (int(payload["bidirectional"]) for payload in payloads), default=None
        ),
        "payload_per_client_max_bytes": max(
            (int(payload["bidirectional"]) for payload in payloads), default=None
        ),
        "logical_payload_total_bytes": total_payload,
        "four_s_reference_total_bytes": total_four_s,
        "logical_payload_reduction_fraction": (
            1.0 - total_payload / total_four_s if total_four_s else None
        ),
        "wallclock_seconds": wallclock_seconds(seed_root, complete),
        "results_csv": str(result_path),
        "replay_diagnostics": str(diagnostic_path),
    }


def direct_seed_record(
    direct_root: Path, dataset: str, mode: str, seed: int
) -> dict[str, Any]:
    seed_root = direct_seed_root(direct_root, dataset, mode, seed)
    result_path, rows = read_result_rows(seed_root)
    complete = (seed_root / "COMPLETED.json").is_file()
    return {
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        **curve_summary(rows, TARGETS[dataset]),
        "wallclock_seconds": wallclock_seconds(seed_root, complete),
        "results_csv": str(result_path),
    }


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def aggregate_group(
    exact: list[dict[str, Any]], direct: list[dict[str, Any]]
) -> dict[str, Any]:
    exact = sorted(exact, key=lambda row: row["seed"])
    direct = sorted(direct, key=lambda row: row["seed"])
    paired_delta = [
        exact_row["final_accuracy"] - direct_row["final_accuracy"]
        for exact_row, direct_row in zip(exact, direct)
    ]
    target_rounds = [
        float(row["rounds_to_target"])
        for row in exact
        if row["rounds_to_target"] is not None
    ]
    wallclocks = [
        float(row["wallclock_seconds"])
        for row in exact
        if row["wallclock_seconds"] is not None
    ]
    direct_wallclocks = [
        float(row["wallclock_seconds"])
        for row in direct
        if row["wallclock_seconds"] is not None
    ]
    payload = sum(int(row["logical_payload_total_bytes"]) for row in exact)
    four_s_payload = sum(int(row["four_s_reference_total_bytes"]) for row in exact)
    return {
        "dataset": exact[0]["dataset"],
        "mode": exact[0]["mode"],
        "complete_seeds": sum(bool(row["complete"]) for row in exact),
        "exact_final_accuracy": mean_std(
            [float(row["final_accuracy"]) for row in exact]
        ),
        "direct_final_accuracy": mean_std(
            [float(row["final_accuracy"]) for row in direct]
        ),
        "paired_final_delta": mean_std(paired_delta),
        "exact_best_accuracy": mean_std(
            [float(row["best_accuracy"]) for row in exact]
        ),
        "direct_best_accuracy": mean_std(
            [float(row["best_accuracy"]) for row in direct]
        ),
        "exact_final_loss": mean_std([float(row["final_loss"]) for row in exact]),
        "rounds_to_target": mean_std(target_rounds) if target_rounds else None,
        "target_reached_seeds": len(target_rounds),
        "wallclock_seconds": mean_std(wallclocks) if wallclocks else None,
        "direct_wallclock_seconds": (
            mean_std(direct_wallclocks) if direct_wallclocks else None
        ),
        "runtime_ratio": (
            statistics.fmean(wallclocks) / statistics.fmean(direct_wallclocks)
            if wallclocks and direct_wallclocks
            else None
        ),
        "logical_payload_total_bytes": payload,
        "four_s_reference_total_bytes": four_s_payload,
        "logical_payload_reduction_fraction": (
            1.0 - payload / four_s_payload if four_s_payload else None
        ),
        "max_parameter_replay_error": max(
            float(row["parameter_max_abs_error"] or 0.0) for row in exact
        ),
        "max_momentum_replay_error": max(
            float(row["momentum_max_abs_error"] or 0.0) for row in exact
        ),
    }


def audit_matrix(args: argparse.Namespace) -> dict[str, Any]:
    exact_records: list[dict[str, Any]] = []
    direct_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in DATASETS:
        for mode in MODES:
            for seed in SEEDS:
                label = f"{dataset}/{mode}/seed-{seed}"
                try:
                    exact_records.append(
                        exact_seed_record(args.replay_root, dataset, mode, seed)
                    )
                except (FileNotFoundError, ValueError) as error:
                    missing.append(f"exact {label}: {error}")
                try:
                    direct_records.append(
                        direct_seed_record(args.direct_root, dataset, mode, seed)
                    )
                except (FileNotFoundError, ValueError) as error:
                    missing.append(f"Direct {label}: {error}")

    aggregates: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for mode in MODES:
            exact_group = [
                row
                for row in exact_records
                if row["dataset"] == dataset and row["mode"] == mode
            ]
            direct_group = [
                row
                for row in direct_records
                if row["dataset"] == dataset and row["mode"] == mode
            ]
            if (
                len(exact_group) == len(SEEDS)
                and len(direct_group) == len(SEEDS)
                and all(bool(row["complete"]) for row in exact_group)
            ):
                aggregates.append(aggregate_group(exact_group, direct_group))

    incomplete = [
        f"{row['dataset']}/{row['mode']}/seed-{row['seed']}"
        for row in exact_records
        if not row["complete"]
    ]
    trace_count = sum(int(row["diagnostic_traces"]) for row in exact_records)
    max_parameter_error = max(
        (float(row["parameter_max_abs_error"] or 0.0) for row in exact_records),
        default=None,
    )
    max_momentum_error = max(
        (float(row["momentum_max_abs_error"] or 0.0) for row in exact_records),
        default=None,
    )
    passed = not missing and not incomplete and len(exact_records) == 12
    return {
        "passed": passed,
        "expected_runs": 12,
        "discovered_exact_runs": len(exact_records),
        "complete_exact_runs": sum(bool(row["complete"]) for row in exact_records),
        "diagnostic_traces": trace_count,
        "expected_diagnostic_traces": 12 * EXPECTED_ROUNDS * CLIENTS_PER_ROUND,
        "max_parameter_replay_error": max_parameter_error,
        "max_momentum_replay_error": max_momentum_error,
        "missing_or_invalid": missing,
        "incomplete": incomplete,
        "per_seed": exact_records,
        "direct_per_seed": direct_records,
        "aggregate": aggregates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_matrix(args)
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(
        f"exact replay: {audit['complete_exact_runs']}/{audit['expected_runs']} "
        f"complete, {audit['diagnostic_traces']}/"
        f"{audit['expected_diagnostic_traces']} traces"
    )
    for row in audit["aggregate"]:
        final = row["exact_final_accuracy"]
        delta = row["paired_final_delta"]
        reduction = 100.0 * row["logical_payload_reduction_fraction"]
        print(
            f"{row['dataset']}/{row['mode']}: "
            f"final={100 * final['mean']:.2f}% +/- "
            f"{100 * float(final['sample_std']):.2f} pp, "
            f"paired delta={100 * delta['mean']:+.2f} pp, "
            f"payload reduction={reduction:.2f}%"
        )
    if audit["missing_or_invalid"]:
        print("missing/invalid:")
        for item in audit["missing_or_invalid"]:
            print(f"  - {item}")
    if audit["incomplete"]:
        print("incomplete:")
        for item in audit["incomplete"]:
            print(f"  - {item}")
    return 0 if audit["passed"] or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
