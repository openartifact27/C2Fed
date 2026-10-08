import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.paper.summarize_client_parallel import (
    main as summarize,
    max_state_error,
    overall_grade,
    summarize_dataset,
)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def create_run(
    root, dataset, workers, repeat, latency, state_value, deterministic=False
):
    run = root / dataset / "epoch" / f"workers-{workers}" / f"repeat-{repeat}"
    run.mkdir(parents=True)
    manifest = {
        "dataset": dataset,
        "local_policy": "epoch",
        "base_seed": 0,
        "warmup_rounds": 0,
        "measure_rounds": 2,
        "repeat_id": repeat,
        "num_workers": workers,
        "gpus": list(range(workers)),
        "partition_fingerprint": "partition",
        "payload_ledger": {"bidirectional": 400},
        "git": {"commit": "abc", "dirty": False},
        "training_args": {
            "dataset": dataset,
            "batch_num": 0,
            "logdir": str(run),
            "device": "cpu",
            "deterministic": deterministic,
        },
    }
    write_json(run / "manifest.json", manifest)
    write_json(run / "COMPLETED.json", {"status": "complete"})
    rows = []
    for round_id in range(2):
        rows.append(
            {
                "round": round_id,
                "measured": True,
                "selected_clients": [0, 1, 2, 3, 4],
                "local_steps": [5, 5, 5, 5, 5],
                "round_latency_seconds": latency,
                "worker_span_seconds": latency * 0.9,
                "aggregation_seconds": 0.01,
                "worker_peak_memory_bytes": [100 for _ in range(workers)],
            }
        )
    (run / "round_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    write_json(
        run / "gpu_utilization.json",
        {
            "error": None,
            "samples": [
                {
                    "gpus": [
                        {"utilization_percent": 80, "index": index}
                        for index in range(workers)
                    ]
                }
            ],
        },
    )
    torch.save(
        {
            "parameters": {"head.weight": torch.tensor([state_value])},
            "momentum": {0: torch.tensor([2.0])},
        },
        run / "final_state.pt",
    )


class ParallelSummaryTest(unittest.TestCase):
    def test_summary_can_leave_historical_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            for workers in (1, 2):
                create_run(root, "cifar100", workers, 0, 10.0 / workers, 1.0)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = Path(directory) / "output"
            self.assertEqual(summarize([
                "--root", str(root), "--datasets", "cifar100", "--repeats", "0",
                "--output-root", str(output),
            ]), 0)
            self.assertTrue((output / "parallel_summary.json").is_file())
            after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_max_state_error(self):
        first = {
            "parameters": {"x": torch.tensor([1.0])},
            "momentum": {0: torch.tensor([2.0])},
        }
        second = {
            "parameters": {"x": torch.tensor([1.00001])},
            "momentum": {0: torch.tensor([2.0])},
        }
        self.assertAlmostEqual(max_state_error(first, second), 1e-5, places=6)

    def test_dataset_summary_grades_strong_scaling_green(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat in (0, 1, 2):
                create_run(root, "cifar100", 1, repeat, 10.0, 1.0)
                create_run(root, "cifar100", 2, repeat, 6.0, 1.0)
            summary = summarize_dataset(root, "cifar100", "epoch", (0, 1, 2), 1e-6)
        self.assertEqual(summary["grade"], "green")
        self.assertAlmostEqual(summary["mean_speedup"], 10.0 / 6.0)
        self.assertTrue(summary["correct"])

    def test_state_mismatch_is_red(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_run(root, "tinyimagenet", 1, 0, 10.0, 1.0)
            create_run(root, "tinyimagenet", 2, 0, 6.0, 1.1)
            summary = summarize_dataset(root, "tinyimagenet", "epoch", (0,), 1e-6)
        self.assertEqual(summary["grade"], "red")
        self.assertFalse(summary["correct"])

    def test_native_timing_uses_separate_deterministic_correctness_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "timing"
            audit = Path(directory) / "audit"
            for repeat in (0, 1, 2):
                create_run(root, "cifar100", 1, repeat, 10.0, 1.0)
                create_run(root, "cifar100", 2, repeat, 6.0, 1.1)
            create_run(audit, "cifar100", 1, 0, 10.0, 1.0, deterministic=True)
            create_run(audit, "cifar100", 2, 0, 6.0, 1.0, deterministic=True)
            summary = summarize_dataset(
                root,
                "cifar100",
                "epoch",
                (0, 1, 2),
                1e-6,
                correctness_audit_root=audit,
            )
        self.assertEqual(summary["grade"], "green")
        self.assertTrue(summary["correct"])
        self.assertEqual(summary["max_state_error"], 0.0)
        self.assertGreater(summary["timing_max_state_error"], 0.0)

    def test_overall_grade(self):
        self.assertEqual(overall_grade(({"grade": "green"}, {"grade": "green"})), "green")
        self.assertEqual(overall_grade(({"grade": "green"}, {"grade": "yellow"})), "yellow")
        self.assertEqual(overall_grade(({"grade": "green"}, {"grade": "red"})), "red")


if __name__ == "__main__":
    unittest.main()
