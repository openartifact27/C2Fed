import tempfile
import unittest
from pathlib import Path

from experiments.paper.summarize_three_seeds import aggregate_records, main as summarize
from experiments.paper.three_seed_runner import (
    Job,
    build_job,
    parse_args,
    parse_gpu_memory_csv,
    prepare_attempt,
)


class ThreeSeedExperimentTest(unittest.TestCase):
    def test_summary_can_leave_historical_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "input/direct/epoch/cifar100/seed-0"
            run.mkdir(parents=True)
            (run / "COMPLETED.json").write_text("{}\n")
            (run / "results-test.csv").write_text(
                "round,test_avg_acc\n0,0.5\n1,0.6\n"
            )
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = root / "output"
            self.assertEqual(summarize([
                "--input-root", str(root / "input"), "--output-root", str(output)
            ]), 0)
            self.assertTrue((output / "aggregate_summary.json").is_file())
            after = {p: p.read_bytes() for p in (root / "input").rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_parse_gpu_memory_csv(self):
        output = "0, 123\n1, 456\nmalformed\n2, N/A\n"
        self.assertEqual(parse_gpu_memory_csv(output), {"0": 123, "1": 456})

    def test_prepare_attempt_rotates_previous_log_and_clears_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            run_root.mkdir()
            job = Job(
                dataset="cifar100",
                seed=0,
                protocol="direct",
                local_policy="epoch",
                run_root=run_root,
                command=("python3", "c2fed.py"),
            )
            job.log_path.write_text("old attempt\n", encoding="utf-8")
            (run_root / "FAILED.json").write_text("{}\n", encoding="utf-8")

            prepare_attempt(job, "0")

            self.assertFalse(job.log_path.exists())
            self.assertFalse((run_root / "FAILED.json").exists())
            archived_logs = list(run_root.glob("stdout.previous-*.log"))
            self.assertEqual(len(archived_logs), 1)
            self.assertEqual(
                archived_logs[0].read_text(encoding="utf-8"), "old attempt\n"
            )
            self.assertTrue((run_root / "launch_manifest.json").exists())

    def test_default_protocol_is_direct(self):
        args = parse_args(["--gpus", "0"])
        self.assertEqual(args.protocol, "direct")
        self.assertEqual(args.local_policy, "epoch")

    def test_build_job_isolates_seed_and_uses_direct_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = build_job(
                dataset="cifar100",
                seed=2,
                protocol="direct",
                local_policy="epoch",
                comm_rounds=100,
                test_freq=5,
                data_dir=root / "data",
                output_root=root / "results",
            )

        self.assertEqual(job.run_root.name, "seed-2")
        self.assertIn("--init_seed", job.command)
        self.assertEqual(job.command[job.command.index("--init_seed") + 1], "2")
        self.assertEqual(
            job.command[job.command.index("--momentum_type") + 1], "direct"
        )
        self.assertEqual(job.command[job.command.index("--clip_norm") + 1], "1")
        self.assertEqual(
            job.command[job.command.index("--config_profile") + 1], "c2fed-direct"
        )
        self.assertEqual(job.command[job.command.index("-K") + 1], "0")
        self.assertEqual(job.command[job.command.index("--test_round") + 1], "99")

    def test_aggregate_uses_sample_standard_deviation(self):
        records = [
            {
                "dataset": "cifar100",
                "seed": seed,
                "final_accuracy": accuracy,
                "best_accuracy": accuracy + 0.01,
                "rounds_to_target": 40 + seed,
                "target_accuracy": 0.85,
            }
            for seed, accuracy in enumerate((0.86, 0.88, 0.90))
        ]

        aggregate = aggregate_records(records)[0]

        self.assertEqual(aggregate["num_seeds"], 3)
        self.assertAlmostEqual(aggregate["final_accuracy_mean"], 0.88)
        self.assertAlmostEqual(aggregate["final_accuracy_sample_std"], 0.02)
        self.assertEqual(aggregate["rounds_to_target_mean"], 41.0)

    def test_k5_uses_the_same_direct_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = build_job(
                dataset="cifar100",
                seed=0,
                protocol="direct",
                local_policy="k5",
                comm_rounds=5,
                test_freq=5,
                data_dir=root / "data",
                output_root=root / "results",
            )

        self.assertEqual(
            job.command[job.command.index("--config_profile") + 1], "c2fed-direct"
        )
        self.assertEqual(job.command[job.command.index("-K") + 1], "5")
        self.assertEqual(job.command[job.command.index("--clip_norm") + 1], "1")


if __name__ == "__main__":
    unittest.main()
