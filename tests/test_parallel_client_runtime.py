import time
import unittest

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


class EchoHandler:
    def __init__(self, worker_id, device, config):
        self.worker_id = worker_id
        self.device = device
        self.fail_client = config.get("fail_client")

    def run_round(self, task):
        start = time.monotonic()
        results = []
        for client_id in task.client_ids:
            if client_id == self.fail_client:
                raise RuntimeError(f"requested failure for client {client_id}")
            results.append(
                ClientResult(
                    round_id=task.round_id,
                    client_id=client_id,
                    final_parameters={"seed": task.task_seeds[client_id]},
                    final_momentum={"worker": self.worker_id},
                    local_steps=task.expected_local_steps[client_id],
                    cuda_compute_seconds=0.0,
                    wall_seconds=0.0,
                    peak_memory_bytes=0,
                )
            )
        elapsed = time.monotonic() - start
        return WorkerReport(
            worker_id=self.worker_id,
            round_id=task.round_id,
            client_results=tuple(results),
            worker_wall_seconds=elapsed,
            worker_cuda_seconds=0.0,
            peak_memory_bytes=0,
        )


def build_echo_handler(worker_id, device, config):
    return EchoHandler(worker_id, device, config)


def make_task(round_id, clients):
    clients = tuple(clients)
    return RoundTask(
        round_id=round_id,
        global_parameters={"value": 1},
        global_momentum={"value": 2},
        client_ids=clients,
        task_seeds={client_id: derive_task_seed(7, round_id, client_id) for client_id in clients},
        expected_local_steps={client_id: client_id + 1 for client_id in clients},
    )


class SchedulingTest(unittest.TestCase):
    def test_task_seed_is_stable_and_task_specific(self):
        seed = derive_task_seed(7, 3, 11)
        self.assertEqual(seed, derive_task_seed(7, 3, 11))
        self.assertNotEqual(seed, derive_task_seed(7, 4, 11))
        self.assertNotEqual(seed, derive_task_seed(7, 3, 12))

    def test_equal_work_schedules_five_clients_as_three_plus_two(self):
        clients = (10, 11, 12, 13, 14)
        schedule = deterministic_greedy_schedule(
            clients, {client_id: 5 for client_id in clients}, num_workers=2
        )
        self.assertEqual(schedule, ((10, 12, 14), (11, 13)))

    def test_greedy_schedule_balances_unequal_work(self):
        schedule = deterministic_greedy_schedule(
            (0, 1, 2, 3), {0: 10, 1: 8, 2: 4, 3: 2}, num_workers=2
        )
        loads = [sum({0: 10, 1: 8, 2: 4, 3: 2}[client] for client in group) for group in schedule]
        self.assertEqual(loads, [12, 12])

    def test_schedule_rejects_incomplete_work_map(self):
        with self.assertRaisesRegex(ValueError, "exactly client_ids"):
            deterministic_greedy_schedule((0, 1), {0: 1}, num_workers=2)


class ResultOrderingTest(unittest.TestCase):
    def test_results_restore_selected_client_order(self):
        report0 = WorkerReport(
            worker_id=0,
            round_id=2,
            client_results=(
                ClientResult(2, 12, {}, {}, 1, 0.0, 0.0, 0),
                ClientResult(2, 14, {}, {}, 1, 0.0, 0.0, 0),
            ),
            worker_wall_seconds=0.0,
            worker_cuda_seconds=0.0,
            peak_memory_bytes=0,
        )
        report1 = WorkerReport(
            worker_id=1,
            round_id=2,
            client_results=(
                ClientResult(2, 11, {}, {}, 1, 0.0, 0.0, 0),
                ClientResult(2, 13, {}, {}, 1, 0.0, 0.0, 0),
            ),
            worker_wall_seconds=0.0,
            worker_cuda_seconds=0.0,
            peak_memory_bytes=0,
        )
        ordered = order_client_results((report0, report1), (11, 12, 13, 14))
        self.assertEqual([result.client_id for result in ordered], [11, 12, 13, 14])

    def test_results_reject_missing_client(self):
        report = WorkerReport(0, 0, (), 0.0, 0.0, 0)
        with self.assertRaisesRegex(ValueError, "missing"):
            order_client_results((report,), (0,))


class PersistentPoolTest(unittest.TestCase):
    def test_spawned_pool_runs_two_rounds(self):
        specs = (WorkerSpec(0, "cpu:0"), WorkerSpec(1, "cpu:1"))
        with ParallelClientPool(
            specs,
            "tests.test_parallel_client_runtime:build_echo_handler",
            startup_timeout_seconds=20,
            round_timeout_seconds=20,
        ) as pool:
            reports0 = pool.run_round({0: make_task(0, (0, 2, 4)), 1: make_task(0, (1, 3))})
            reports1 = pool.run_round({0: make_task(1, (0, 2, 4)), 1: make_task(1, (1, 3))})
        self.assertEqual([report.worker_id for report in reports0], [0, 1])
        self.assertEqual([report.worker_id for report in reports1], [0, 1])
        ordered = order_client_results(reports1, (0, 1, 2, 3, 4))
        self.assertEqual([result.client_id for result in ordered], [0, 1, 2, 3, 4])

    def test_worker_failure_is_propagated(self):
        specs = (WorkerSpec(0, "cpu:0"),)
        pool = ParallelClientPool(
            specs,
            "tests.test_parallel_client_runtime:build_echo_handler",
            {"fail_client": 3},
            startup_timeout_seconds=20,
            round_timeout_seconds=20,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "requested failure"):
                pool.run_round({0: make_task(0, (3,))})
        finally:
            pool.close(force=True)


if __name__ == "__main__":
    unittest.main()
