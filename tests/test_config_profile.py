import unittest
from argparse import Namespace

from c2fed import validate_experiment_profile


def main_profile(**overrides):
    values = {
        "config_profile": "c2fed-direct",
        "train": "fwd",
        "fwd_head_tuning": True,
        "batch_num": 5,
        "perturb_num": 8,
        "micro_batch_size": 8,
        "num_micro_batch": 2,
        "sample_selection": "uncertainty",
        "h": 0.001,
        "clip_norm": 1.0,
        "momentum_type": "direct",
        "momentum": 0.9,
    }
    values.update(overrides)
    return Namespace(**values)


class ConfigProfileTest(unittest.TestCase):
    def test_accepts_exact_paper_aligned_configuration(self):
        summary = validate_experiment_profile(main_profile())

        self.assertEqual(summary["selected_samples"], 16)
        self.assertEqual(summary["communication_ledger"], "4S")
        self.assertFalse(summary["replay_performance_mode"])

    def test_replay_accepts_k5_and_full_epoch_in_both_execution_modes(self):
        for batches in (0, 5):
            for performance in (False, True):
                with self.subTest(batches=batches, performance=performance):
                    summary = validate_experiment_profile(main_profile(
                        config_profile="c2fed-replay",
                        momentum_type="replay-ht",
                        batch_num=batches,
                        replay_performance_mode=performance,
                    ))
                    self.assertEqual(summary["communication_ledger"], "2S+2H+Q(T)")
                    self.assertEqual(summary["replay_performance_mode"], performance)

    def test_main_profile_accepts_full_epoch_and_k5(self):
        for batches in (0, 5):
            validate_experiment_profile(main_profile(batch_num=batches))

    def test_replay_profile_rejects_configuration_drift(self):
        for override in (
            {"momentum_type": "direct"}, {"batch_num": 7},
            {"fwd_head_tuning": False}, {"clip_norm": 0.0},
        ):
            with self.subTest(override=override):
                config = main_profile(
                    config_profile="c2fed-replay", momentum_type="replay-ht"
                )
                vars(config).update(override)
                with self.assertRaisesRegex(ValueError, "c2fed-replay configuration mismatch"):
                    validate_experiment_profile(config)

    def test_rejects_missing_clipping(self):
        with self.assertRaisesRegex(ValueError, "--clip_norm"):
            validate_experiment_profile(main_profile(clip_norm=0.0))

    def test_rejects_replay_as_main_profile(self):
        with self.assertRaisesRegex(ValueError, "--momentum_type"):
            validate_experiment_profile(main_profile(momentum_type="replay-ht"))


if __name__ == "__main__":
    unittest.main()
