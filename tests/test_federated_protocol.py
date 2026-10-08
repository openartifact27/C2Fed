import unittest

import torch

from federated_protocol import (
    direct_payload_bytes,
    derive_replay_seed,
    is_head_parameter,
    named_momentum_from_optimizer_state,
    optimizer_state_from_named_momentum,
    replay_ht_payload_bytes,
    replay_peft_trace,
    split_head_parameters,
)


class FederatedProtocolTest(unittest.TestCase):
    def setUp(self):
        self.global_state = {
            "_orig_mod.encoder.bias": torch.tensor([10.0, 20.0]),
            "_orig_mod.head.weight": torch.tensor([30.0]),
        }
        self.client_states = {
            2: {
                "_orig_mod.encoder.bias": torch.tensor([8.0, 18.0]),
                "_orig_mod.head.weight": torch.tensor([27.0]),
            },
            7: {
                "_orig_mod.encoder.bias": torch.tensor([6.0, 16.0]),
                "_orig_mod.head.weight": torch.tensor([25.0]),
            },
        }

    def test_head_parameter_detection_handles_compiled_names(self):
        self.assertTrue(is_head_parameter("head.weight"))
        self.assertTrue(is_head_parameter("_orig_mod.head.bias"))
        self.assertFalse(is_head_parameter("_orig_mod.encoder.layer.0.bias"))

        peft, head = split_head_parameters(self.global_state)
        self.assertEqual(list(peft), ["_orig_mod.encoder.bias"])
        self.assertEqual(list(head), ["_orig_mod.head.weight"])

    def test_named_momentum_can_decode_only_selected_parameters(self):
        state = {
            "state": {
                0: {"momentum_buffer": torch.tensor([1.0])},
                1: {"momentum_buffer": torch.tensor([2.0])},
            }
        }
        decoded = named_momentum_from_optimizer_state(
            state,
            ["encoder.bias", "head.bias"],
            selected_names={"head.bias"},
        )
        self.assertEqual(list(decoded), ["head.bias"])
        torch.testing.assert_close(decoded["head.bias"], torch.tensor([2.0]))

    def test_direct_ledger_is_four_times_compact_state(self):
        ledger = direct_payload_bytes(self.global_state)
        self.assertEqual(ledger["downlink"], 2 * ledger["compact_state"])
        self.assertEqual(ledger["uplink"], 2 * ledger["compact_state"])
        self.assertEqual(ledger["bidirectional"], 4 * ledger["compact_state"])

    def test_replay_payload_ledger_matches_exact_formula(self):
        _, head = split_head_parameters(self.global_state)
        ledger = replay_ht_payload_bytes(
            self.global_state,
            head,
            local_steps=5,
            jvps_per_step=16,
        )
        self.assertEqual(ledger["seed_bytes"], 5 * 8)
        self.assertEqual(ledger["jvp_bytes"], 5 * 16 * 4)
        self.assertEqual(ledger["downlink"], 2 * ledger["compact_state"])
        self.assertEqual(
            ledger["uplink"],
            2 * ledger["head_state"] + ledger["seed_bytes"] + ledger["jvp_bytes"],
        )
        self.assertEqual(
            ledger["bidirectional"],
            2 * ledger["compact_state"]
            + 2 * ledger["head_state"]
            + ledger["seed_bytes"]
            + ledger["jvp_bytes"],
        )

    def test_replay_seed_is_stable_and_coordinate_specific(self):
        seed = derive_replay_seed(7, 11, 13, 17)
        self.assertEqual(seed, derive_replay_seed(7, 11, 13, 17))
        self.assertNotEqual(seed, derive_replay_seed(7, 11, 13, 18))

    def test_multi_step_replay_recovers_parameters_and_momentum(self):
        initial = {
            "encoder.a": torch.tensor([1.0, -2.0]),
            "encoder.b": torch.tensor([[0.5, -0.25], [0.75, 1.25]]),
        }
        initial_momentum = {
            name: torch.full_like(value, 0.125)
            for name, value in initial.items()
        }
        learning_rate = 0.1
        momentum = 0.9
        smooth_factor = 0.001
        clip_norm = 1.0

        for local_steps in (1, 5, 7):
            with self.subTest(local_steps=local_steps):
                parameters = {
                    name: torch.nn.Parameter(value.clone())
                    for name, value in initial.items()
                }
                ordered = list(parameters.values())
                optimizer = torch.optim.SGD(
                    ordered,
                    lr=learning_rate,
                    momentum=momentum,
                )
                for name, parameter in parameters.items():
                    optimizer.state[parameter]["momentum_buffer"] = initial_momentum[name].clone()

                trace = []
                for step in range(local_steps):
                    seed = derive_replay_seed(3, 4, 5, step)
                    generator = torch.Generator().manual_seed(seed)
                    masks = [torch.rand(p.shape, generator=generator) > 0 for p in ordered]
                    for parameter in ordered:
                        parameter.grad = torch.zeros_like(parameter)
                    jvps = torch.tensor([0.25 + step, -0.5, 1.25], dtype=torch.float32)
                    for scalar in jvps:
                        for parameter, mask in zip(ordered, masks):
                            perturbation = torch.randn(
                                parameter.shape,
                                generator=generator,
                            ).mul_(mask).mul_(smooth_factor)
                            parameter.grad.add_(perturbation.mul_(scalar))
                    torch.nn.utils.clip_grad_norm_(ordered, clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=False)
                    trace.append({"seed": seed, "jvps": jvps})

                replayed_parameters, replayed_momentum = replay_peft_trace(
                    initial,
                    initial_momentum,
                    trace,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    dampening=0,
                    weight_decay=0,
                    smooth_factor=smooth_factor,
                    mask_ratio=0,
                    clip_norm=clip_norm,
                )
                for name, parameter in parameters.items():
                    torch.testing.assert_close(
                        replayed_parameters[name], parameter, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        replayed_momentum[name],
                        optimizer.state[parameter]["momentum_buffer"],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
