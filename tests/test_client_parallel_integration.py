import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from experiments.paper.client_parallel_runner import (
    PREWARM_ROUND_ID,
    aggregate_client_results,
    build_round_tasks,
    build_prewarm_tasks,
    build_training_args,
    local_steps_from_sizes,
    partition_fingerprint,
    seed_everything,
)
from parallel_client_runtime import ClientResult


def client_result(client_id, parameter_value, momentum_value):
    return ClientResult(
        round_id=0,
        client_id=client_id,
        final_parameters={"head.weight": torch.tensor([parameter_value])},
        final_momentum={0: torch.tensor([momentum_value])},
        local_steps=5,
        cuda_compute_seconds=0.0,
        wall_seconds=0.0,
        peak_memory_bytes=0,
    )


class ParallelRunnerConfigurationTest(unittest.TestCase):
    def test_task_seed_accepts_full_pytorch_seed_range(self):
        seed_everything((1 << 63) - 1, include_cuda=False)

    def test_primary_epoch_configuration_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_training_args(
                "cifar100", Path(directory), Path(directory), 0, "epoch"
            )
        self.assertEqual(args.batch_num, 0)
        self.assertEqual(args.peftmode, "bitfit")
        self.assertEqual(args.momentum_type, "direct")
        self.assertEqual(args.clip_norm, 1.0)
        self.assertEqual(args.h, 0.001)
        self.assertEqual(args.perturb_num, 8)
        self.assertEqual(args.micro_batch_size * args.num_micro_batch, 16)
        self.assertFalse(args.deterministic)

    def test_deterministic_mode_is_explicit_audit_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_training_args(
                "cifar100",
                Path(directory),
                Path(directory),
                0,
                "epoch",
                deterministic=True,
            )
        self.assertTrue(args.deterministic)

    def test_k5_only_changes_local_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epoch = build_training_args("tinyimagenet", root, root, 3, "epoch")
            k5 = build_training_args("tinyimagenet", root, root, 3, "k5")
        epoch_values = vars(epoch).copy()
        k5_values = vars(k5).copy()
        self.assertEqual(epoch_values.pop("batch_num"), 0)
        self.assertEqual(k5_values.pop("batch_num"), 5)
        self.assertEqual(epoch_values, k5_values)

    def test_prewarm_tasks_compile_each_worker_before_real_rounds(self):
        parameters = OrderedDict(weight=torch.tensor([1.0]))
        tasks = build_prewarm_tasks(
            num_workers=2,
            parameters=parameters,
            local_steps={0: 7},
            base_seed=11,
        )
        self.assertEqual(set(tasks), {0, 1})
        self.assertEqual(tasks[0].round_id, PREWARM_ROUND_ID)
        self.assertEqual(tasks[0].client_ids, (0,))
        self.assertEqual(tasks[0].task_seeds, tasks[1].task_seeds)
        self.assertIs(tasks[0].global_parameters, parameters)
        self.assertEqual(tasks[0].global_momentum, {})

    def test_local_step_count_matches_drop_last_loader(self):
        index_map = {0: torch.arange(100).numpy(), 1: torch.arange(960).numpy()}
        self.assertEqual(local_steps_from_sizes(index_map, 64, 0), {0: 1, 1: 15})
        self.assertEqual(local_steps_from_sizes(index_map, 64, 5), {0: 5, 1: 5})

    def test_partition_fingerprint_is_order_independent_by_mapping_order(self):
        first = {0: torch.tensor([1, 2]).numpy(), 1: torch.tensor([3]).numpy()}
        second = {1: torch.tensor([3]).numpy(), 0: torch.tensor([1, 2]).numpy()}
        self.assertEqual(partition_fingerprint(first), partition_fingerprint(second))


class ParallelAggregationTest(unittest.TestCase):
    def test_parameters_and_client_momentum_use_equal_mean(self):
        parameters, momentum = aggregate_client_results(
            (client_result(4, 1.0, 3.0), client_result(8, 5.0, 7.0))
        )
        torch.testing.assert_close(parameters["head.weight"], torch.tensor([3.0]))
        torch.testing.assert_close(momentum[0], torch.tensor([5.0]))

    def test_round_tasks_cover_schedule_and_share_state(self):
        parameters = {"head.weight": torch.tensor([1.0])}
        momentum = {0: torch.tensor([2.0])}
        tasks = build_round_tasks(
            round_id=2,
            selected_clients=(10, 11, 12, 13, 14),
            schedule=((10, 12, 14), (11, 13)),
            parameters=parameters,
            momentum=momentum,
            local_steps={client_id: 5 for client_id in range(20)},
            base_seed=7,
        )
        self.assertEqual(tasks[0].client_ids, (10, 12, 14))
        self.assertEqual(tasks[1].client_ids, (11, 13))
        self.assertIs(tasks[0].global_parameters, parameters)
        self.assertIs(tasks[1].global_momentum, momentum)
        self.assertNotEqual(tasks[0].task_seeds[10], tasks[1].task_seeds[11])

    def test_round_tasks_reject_missing_client(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            build_round_tasks(
                0,
                (1, 2),
                ((1,), ()),
                {"x": torch.tensor([0.0])},
                {},
                {1: 1, 2: 1},
                0,
            )


if __name__ == "__main__":
    unittest.main()
