"""Federated state and payload helpers for C2Fed.

The functions in this module are deliberately independent of datasets and
models so that Direct and Exact Replay can be tested without a full run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib

import torch


TensorState = Mapping[str, torch.Tensor]


def is_head_parameter(name: str) -> bool:
    """Return whether ``name`` belongs to the classification head."""

    return "head" in name.split(".")


def split_head_parameters(parameters: Mapping[str, torch.Tensor]):
    """Split a named parameter mapping into PEFT/backbone and head groups."""

    peft = {name: value for name, value in parameters.items() if not is_head_parameter(name)}
    head = {name: value for name, value in parameters.items() if is_head_parameter(name)}
    return peft, head


def optimizer_state_from_named_momentum(
    shared_momentum: TensorState,
    parameter_order: Iterable[str],
    base_optimizer_state: Mapping,
) -> dict:
    """Encode named shared momentum using PyTorch optimizer parameter indices."""

    state = {
        index: {"momentum_buffer": shared_momentum[name].detach().clone()}
        for index, name in enumerate(parameter_order)
    }
    result = {
        "state": state,
        "param_groups": base_optimizer_state["param_groups"],
    }
    return result


def named_momentum_from_optimizer_state(
    optimizer_state: Mapping,
    parameter_order: Sequence[str],
    selected_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Decode PyTorch optimizer indices into stable parameter names."""

    named = {}
    state = optimizer_state.get("state", {})
    for index, name in enumerate(parameter_order):
        if selected_names is not None and name not in selected_names:
            continue
        entry = state.get(index)
        if entry is not None and "momentum_buffer" in entry:
            named[name] = entry["momentum_buffer"].detach().clone()
    return named


def derive_replay_seed(base_seed: int, round_index: int, client_id: int, step: int) -> int:
    """Derive a stable 63-bit seed without consuming application RNG state."""

    payload = f"{base_seed}:{round_index}:{client_id}:{step}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)


@torch.no_grad()
def replay_peft_trace(
    initial_parameters: TensorState,
    initial_momentum: TensorState,
    trace: Sequence[Mapping],
    *,
    learning_rate: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    smooth_factor: float,
    mask_ratio: float,
    clip_norm: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Exactly replay PEFT SGD steps from protocol seeds and JVP scalars."""

    if not initial_parameters:
        raise ValueError("initial_parameters must contain at least one PEFT tensor")
    if learning_rate <= 0 or smooth_factor <= 0:
        raise ValueError("learning_rate and smooth_factor must be positive")

    names = list(initial_parameters)
    parameters = {
        name: torch.nn.Parameter(value.detach().clone(), requires_grad=True)
        for name, value in initial_parameters.items()
    }
    ordered_parameters = [parameters[name] for name in names]
    optimizer = torch.optim.SGD(
        ordered_parameters,
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
    )
    for name, value in initial_momentum.items():
        if name not in parameters:
            raise KeyError(f"momentum provided for unknown PEFT parameter {name!r}")
        optimizer.state[parameters[name]]["momentum_buffer"] = value.detach().clone()

    for step_index, step_trace in enumerate(trace):
        if "seed" not in step_trace or "jvps" not in step_trace:
            raise KeyError(f"trace step {step_index} must contain seed and jvps")
        generator = torch.Generator(device=ordered_parameters[0].device)
        generator.manual_seed(int(step_trace["seed"]))
        # Move the complete step trace in one operation.  The original audit
        # implementation copied every scalar independently, introducing one
        # synchronization per JVP when client and server devices differed.
        step_jvps = step_trace["jvps"].to(
            device=ordered_parameters[0].device,
            dtype=ordered_parameters[0].dtype,
        )
        masks = [
            torch.rand(
                parameter.shape,
                dtype=parameter.dtype,
                device=parameter.device,
                generator=generator,
            ) > mask_ratio
            for parameter in ordered_parameters
        ]
        for parameter in ordered_parameters:
            parameter.grad = torch.zeros_like(parameter)
        for scalar in step_jvps:
            for parameter, mask in zip(ordered_parameters, masks):
                perturbation = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                ).mul_(mask).mul_(smooth_factor)
                parameter.grad.add_(
                    perturbation.mul_(scalar)
                )
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(ordered_parameters, clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

    final_parameters = {
        name: parameters[name].detach().clone()
        for name in names
    }
    final_momentum = {}
    for name in names:
        entry = optimizer.state.get(parameters[name], {})
        if "momentum_buffer" in entry:
            final_momentum[name] = entry["momentum_buffer"].detach().clone()
    return final_parameters, final_momentum


def tensor_state_nbytes(state: TensorState) -> int:
    """Return the logical tensor payload size, excluding serialization metadata."""

    return sum(value.numel() * value.element_size() for value in state.values())


def direct_payload_bytes(compact_state: TensorState) -> dict[str, int]:
    """Return the C2Fed-Direct 2S-down/2S-up/4S-total payload."""

    state_bytes = tensor_state_nbytes(compact_state)
    return {
        "compact_state": state_bytes,
        "downlink": 2 * state_bytes,
        "uplink": 2 * state_bytes,
        "bidirectional": 4 * state_bytes,
    }


def replay_ht_payload_bytes(
    compact_state: TensorState,
    head_state: TensorState,
    *,
    local_steps: int,
    jvps_per_step: int,
) -> dict[str, int]:
    """Return the exact logical ledger for stateless HT/PEFT replay.

    Each local step uploads one uint64 seed and ``jvps_per_step`` float32
    directional-derivative scalars. Head parameters and head momentum are also
    uploaded explicitly; the full parameter and momentum states are sent down.
    """

    if local_steps <= 0 or jvps_per_step <= 0:
        raise ValueError("local_steps and jvps_per_step must be positive")
    state_bytes = tensor_state_nbytes(compact_state)
    head_bytes = tensor_state_nbytes(head_state)
    seed_bytes = local_steps * 8
    jvp_bytes = local_steps * jvps_per_step * 4
    downlink = 2 * state_bytes
    uplink = 2 * head_bytes + seed_bytes + jvp_bytes
    return {
        "compact_state": state_bytes,
        "head_state": head_bytes,
        "local_steps": local_steps,
        "jvps_per_step": jvps_per_step,
        "seed_bytes": seed_bytes,
        "jvp_bytes": jvp_bytes,
        "downlink": downlink,
        "uplink": uplink,
        "bidirectional": downlink + uplink,
        "four_s_reference": 4 * state_bytes,
    }
