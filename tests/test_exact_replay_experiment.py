import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paper.exact_replay_runner import build_command
from c2fed import get_args, validate_experiment_profile


class ExactReplayCommandTest(unittest.TestCase):
    def test_runner_commands_match_the_unified_training_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, expected_steps in (("k5", 5), ("full_epoch", 0)):
                for performance in (False, True):
                    with self.subTest(mode=mode, performance=performance):
                        command = build_command(
                            "cifar100", mode, 0, 2, 1, Path(directory), performance
                        )
                        with patch.object(sys, "argv", command[1:]):
                            args = get_args()
                        summary = validate_experiment_profile(args)
                        self.assertEqual(args.batch_num, expected_steps)
                        self.assertEqual(args.momentum_type, "replay-ht")
                        self.assertEqual(args.replay_performance_mode, performance)
                        self.assertEqual(summary["communication_ledger"], "2S+2H+Q(T)")
                        self.assertEqual(
                            Path(command[1]).resolve(),
                            Path(__file__).resolve().parents[1] / "c2fed.py",
                        )


if __name__ == "__main__":
    unittest.main()
