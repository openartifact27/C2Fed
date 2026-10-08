#!/usr/bin/env python3
"""Summarize C2Fed three-seed CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "artifacts" / "three_seeds"
DEFAULT_TARGETS = {"cifar100": 0.85, "tinyimagenet": 0.84}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, help="Write summaries here without modifying the input artifacts.")
    parser.add_argument("--protocol", choices=("direct",), default="direct")
    parser.add_argument("--local-policy", choices=("epoch", "k5"), default="epoch")
    parser.add_argument("--cifar100-target", type=float, default=0.85)
    parser.add_argument("--tinyimagenet-target", type=float, default=0.84)
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.expanduser().resolve()
    args.targets = {
        "cifar100": args.cifar100_target,
        "tinyimagenet": args.tinyimagenet_target,
    }
    return args


def latest_result_csv(seed_root: Path) -> Path:
    candidates = sorted(
        seed_root.glob("**/results-*.csv"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"no results CSV below {seed_root}")
    return candidates[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"results CSV contains no measurements: {path}")
    return rows


def seed_record(
    dataset: str, seed: int, seed_root: Path, target: float
) -> dict[str, object]:
    csv_path = latest_result_csv(seed_root)
    rows = read_rows(csv_path)
    accuracies = [float(row["test_avg_acc"]) for row in rows]
    rounds = [int(row["round"]) for row in rows]
    target_round = next(
        (
            round_index
            for round_index, accuracy in zip(rounds, accuracies)
            if accuracy >= target
        ),
        None,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "final_accuracy": accuracies[-1],
        "best_accuracy": max(accuracies),
        "rounds_to_target": target_round if target_round is not None else "N/A",
        "target_accuracy": target,
        "last_evaluated_round": rounds[-1],
        "results_csv": str(csv_path),
    }


def discover_records(
    protocol_root: Path, targets: dict[str, float]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset, target in targets.items():
        dataset_root = protocol_root / dataset
        for seed_root in sorted(dataset_root.glob("seed-*")):
            if not (seed_root / "COMPLETED.json").is_file():
                continue
            seed = int(seed_root.name.removeprefix("seed-"))
            records.append(seed_record(dataset, seed, seed_root, target))
    return records


def aggregate_records(
    records: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_dataset.setdefault(str(record["dataset"]), []).append(record)

    aggregate: list[dict[str, object]] = []
    for dataset, dataset_records in sorted(by_dataset.items()):
        dataset_records.sort(key=lambda record: int(record["seed"]))
        final = [float(record["final_accuracy"]) for record in dataset_records]
        best = [float(record["best_accuracy"]) for record in dataset_records]
        target_rounds = [
            int(record["rounds_to_target"])
            for record in dataset_records
            if record["rounds_to_target"] != "N/A"
        ]
        aggregate.append(
            {
                "dataset": dataset,
                "num_seeds": len(dataset_records),
                "seeds": ",".join(str(record["seed"]) for record in dataset_records),
                "final_accuracy_mean": statistics.fmean(final),
                "final_accuracy_sample_std": (
                    statistics.stdev(final) if len(final) > 1 else "N/A"
                ),
                "best_accuracy_mean": statistics.fmean(best),
                "best_accuracy_sample_std": (
                    statistics.stdev(best) if len(best) > 1 else "N/A"
                ),
                "target_accuracy": dataset_records[0]["target_accuracy"],
                "target_reached_seeds": len(target_rounds),
                "rounds_to_target_mean": (
                    statistics.fmean(target_rounds) if target_rounds else "N/A"
                ),
                "rounds_to_target_sample_std": (
                    statistics.stdev(target_rounds)
                    if len(target_rounds) > 1
                    else "N/A"
                ),
            }
        )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_stat(value: object) -> str:
    return value if isinstance(value, str) else f"{float(value):.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_root = args.input_root / args.protocol / args.local_policy
    records = discover_records(protocol_root, args.targets)
    if not records:
        print(f"No completed runs found below {protocol_root}")
        return 1

    aggregate = aggregate_records(records)
    output_root = args.output_root if args.output_root is not None else protocol_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "per_seed_summary.csv", records)
    write_csv(output_root / "aggregate_summary.csv", aggregate)
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    for row in aggregate:
        print(
            f"{row['dataset']}: n={row['num_seeds']}, "
            f"final={format_stat(row['final_accuracy_mean'])} +/- "
            f"{format_stat(row['final_accuracy_sample_std'])}, "
            f"best={format_stat(row['best_accuracy_mean'])} +/- "
            f"{format_stat(row['best_accuracy_sample_std'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
