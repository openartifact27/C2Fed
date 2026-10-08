import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from federated_protocol import derive_replay_seed, replay_peft_trace
from local_training import forward_head_tuning, fwd_train_step


class TinyForwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.peft = torch.nn.Parameter(
            torch.tensor([[1.0, 0.25], [-0.5, 0.75]], dtype=torch.float32)
        )
        self.head = torch.nn.Linear(2, 3, bias=True)

    def forward(self, x):
        cls_token = x @ self.peft
        return {"logits": self.head(cls_token), "cls_token": cls_token}


class AnalyticalHeadGradientTest(unittest.TestCase):
    def test_matches_softmax_cross_entropy_closed_form(self):
        logits = torch.tensor(
            [[1.0, -0.5, 0.25], [0.2, 0.7, -0.1]],
            dtype=torch.float32,
        )
        cls_token = torch.tensor(
            [[1.0, 2.0], [-1.0, 0.5]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 2])

        bias_grad, weight_grad = forward_head_tuning(
            {'logits': logits, 'cls_token': cls_token},
            labels,
        )

        expected = F.softmax(logits, dim=-1)
        expected[torch.arange(labels.numel()), labels] -= 1
        expected /= labels.numel()

        torch.testing.assert_close(bias_grad, expected.sum(dim=0))
        torch.testing.assert_close(weight_grad, expected.T @ cls_token)

    def test_seeded_client_path_matches_five_and_seven_step_replay(self):
        args = SimpleNamespace(
            sample_selection="all",
            micro_batch_size=2,
            num_micro_batch=2,
            fwd_head_tuning=False,
            clip_norm=1.0,
            perturb_num=2,
            h=0.001,
            mask_ratio=0.0,
        )
        x = torch.tensor(
            [[1.0, 0.5], [-0.25, 0.75], [0.2, -1.0], [0.4, 0.3]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 2, 0])

        for local_steps in (5, 7):
            with self.subTest(local_steps=local_steps):
                torch.manual_seed(19)
                model = TinyForwardModel()
                parameter = model.peft
                initial_parameter = {"peft": parameter.detach().clone()}
                initial_momentum = {"peft": torch.full_like(parameter, 0.125)}
                parameter.grad = torch.zeros_like(parameter)
                optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
                optimizer.state[parameter]["momentum_buffer"] = initial_momentum["peft"].clone()
                parameter_data = [parameter.data]
                trace = []

                for step in range(local_steps):
                    step_trace = fwd_train_step(
                        model,
                        x,
                        labels,
                        [parameter],
                        parameter_data,
                        args,
                        replay_seed=derive_replay_seed(2, 3, 4, step),
                    )
                    self.assertEqual(step_trace["jvps"].numel(), 4)
                    trace.append(step_trace)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=False)

                replayed_parameters, replayed_momentum = replay_peft_trace(
                    initial_parameter,
                    initial_momentum,
                    trace,
                    learning_rate=0.1,
                    momentum=0.9,
                    dampening=0,
                    weight_decay=0,
                    smooth_factor=0.001,
                    mask_ratio=0,
                    clip_norm=1.0,
                )
                torch.testing.assert_close(
                    replayed_parameters["peft"], parameter, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    replayed_momentum["peft"],
                    optimizer.state[parameter]["momentum_buffer"],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
