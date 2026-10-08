# 标准库
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from local_training import (
    forward_head_tuning,
    fwd_perturb,
    fwd_train_step,
    generate_random_masks,
    masks_to_one_hot,
    perturb,
    train_local_bp,
    train_local_fwd,
)
from utils import (
    partition_data,
    compute_accuracy,
    fedavg,
    pit,
)
from federated_protocol import (
    direct_payload_bytes,
    derive_replay_seed,
    named_momentum_from_optimizer_state,
    optimizer_state_from_named_momentum,
    replay_ht_payload_bytes,
    replay_peft_trace,
    split_head_parameters,
)
torch.set_float32_matmul_precision('high')


C2FED_DIRECT_PROFILE = "c2fed-direct"
C2FED_REPLAY_PROFILE = "c2fed-replay"


def validate_experiment_profile(args):
    """Reject accidental drift from the evaluated C2Fed configurations."""

    selected_samples = args.micro_batch_size * args.num_micro_batch
    summary = {
        "profile": args.config_profile,
        "local_updates": args.batch_num,
        "perturbations": args.perturb_num,
        "selected_samples": selected_samples,
        "smooth_factor": args.h,
        "clip_norm": args.clip_norm,
        "momentum_type": args.momentum_type,
        "momentum": args.momentum,
        "replay_performance_mode": getattr(args, "replay_performance_mode", False),
        "communication_ledger": (
            "4S" if args.momentum_type == "direct"
            else "2S+2H+Q(T)" if args.momentum_type == "replay-ht"
            else "non-main"
        ),
    }
    if args.config_profile not in {C2FED_DIRECT_PROFILE, C2FED_REPLAY_PROFILE}:
        return summary

    expected = {
        "--train": (args.train, "fwd"),
        "--fwd_head_tuning": (args.fwd_head_tuning, True),
        "--perturb_num/-N": (args.perturb_num, 8),
        "selected samples (micro_batch_size*num_micro_batch)": (selected_samples, 16),
        "--micro_batch_size": (args.micro_batch_size, 8),
        "--num_micro_batch": (args.num_micro_batch, 2),
        "--sample_selection": (args.sample_selection, "uncertainty"),
        "--h": (args.h, 0.001),
        "--clip_norm": (args.clip_norm, 1.0),
        "--momentum": (args.momentum, 0.9),
    }
    allowed_batch_numbers = {0, 5}
    if args.batch_num not in allowed_batch_numbers:
        expected["--batch_num/-K"] = (
            args.batch_num,
            "0 (one local epoch) or 5",
        )
    expected["--momentum_type"] = (
        args.momentum_type,
        "direct" if args.config_profile == C2FED_DIRECT_PROFILE else "replay-ht",
    )
    mismatches = [
        f"{name}: got {actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            f"{args.config_profile} configuration mismatch:\n  "
            + "\n  ".join(mismatches)
        )
    return summary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='roberta-base', help='model name')
    parser.add_argument('--dataset', type=str, default='sst2', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeluni', help='the data partitioning strategy')
    parser.add_argument('--cls_num', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-2)')
    parser.add_argument('--head_lr', type=float, default=0, help='learning rate (default: lr)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=100, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication round')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for optimizier')
    parser.add_argument('--dampening', type=float, default=0, help='dampening for optimizier')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay for optimizier')
    parser.add_argument('--sample', type=float, default=0.05, help='Sample ratio for each communication round')
    parser.add_argument('--test_round', type=int, default=99, help='The round of testing')
    parser.add_argument('--test_freq', type=int, default=5, help='The frequency of testing')
    parser.add_argument('--patience', type=int, default=100, help='The patience of early stopping')
    parser.add_argument('--root_path', type=str, default='', help='Noise type: None/increasng/space')
    parser.add_argument('--check_layer_id', type=int, default=49)
    parser.add_argument('--h', type=float, default=0.001, help='perturbation rate (default: 0.01)')
    parser.add_argument('-N', '--perturb_num', type=int, default=8, help='num of noise in fwd perturb')
    parser.add_argument('-K', '--batch_num', type=int, default=0, help='num of batch in local update')
    parser.add_argument('--micro_batch_size', type=int, default=8, help='input micro-batch size for training (default: 8)')
    parser.add_argument('--num_micro_batch', type=int, default=2, help='input num of micro-batch for training (default: 1)')
    parser.add_argument('--mask_ratio', type=float, default=0, help='mask ratio for fwd perturb')
    parser.add_argument('--train', type=str, default='fwd', choices=['fwd', 'bp', 'hybrid'])
    parser.add_argument('--peftmode', type=str, default='full-ft', choices=['head', 'adapter', 'lora', 'bitfit', 'full-ft'], help='set peft mode')
    parser.add_argument('--adapter_dim', type=int, default=8, help='The dimension of the adapter')
    parser.add_argument('--alpha', type=int, default=8, help='The alpha of the lora')
    parser.add_argument('--r', type=int, default=8, help='The r of the lora')
    parser.add_argument('--fwd_head_tuning', action='store_true')
    parser.add_argument('--partial_layer_training', action='store_true')
    parser.add_argument('--sample_selection', type=str, default='all', choices=['all', 'uncertainty', 'random'], help='sample selection method')
    parser.add_argument(
        '--momentum_type',
        choices=['server', 'direct', 'replay-ht', 'fedavg-m', 'none'],
        default='none',
        help=(
            'Momentum/state protocol. direct exchanges final compact parameters '
            'and client momentum; replay-ht reconstructs PEFT state exactly from '
            'per-step seeds and directional derivatives.'
        ),
    )
    parser.add_argument('--log_grad_norm', action='store_true', help='Whether to log the gradient norm')
    parser.add_argument('--clip_norm', type=float, default=0, help='clip norm')
    parser.add_argument(
        '--replay_performance_mode',
        action='store_true',
        help=(
            'Use the exact replay protocol without per-client full-state '
            'assertions or trajectory JSON. The default remains audit mode.'
        ),
    )
    parser.add_argument(
        '--config_profile',
        choices=['custom', C2FED_DIRECT_PROFILE, C2FED_REPLAY_PROFILE],
        default='custom',
        help='Validate an evaluated C2Fed Direct or Replay configuration',
    )
    return parser.parse_args()

def main():
    args = get_args()
    if args.head_lr == 0:
        args.head_lr = args.lr
    config_summary = validate_experiment_profile(args)
    print("Training configuration: " + json.dumps(config_summary, sort_keys=True))
    if args.momentum_type == 'replay-ht' and args.optimizer != 'sgd':
        raise ValueError("Replay currently supports --optimizer sgd only")
    if args.momentum_type in {'direct', 'replay-ht'} and args.momentum <= 0:
        raise ValueError("C2Fed protocols require a positive --momentum coefficient")
    if args.momentum_type == 'replay-ht' and not args.fwd_head_tuning:
        raise ValueError("--momentum_type replay-ht requires --fwd_head_tuning")
    if args.replay_performance_mode and args.momentum_type != 'replay-ht':
        raise ValueError("--replay_performance_mode requires --momentum_type replay-ht")
    save_path = (f"{args.model_name}-"
                f"{args.dataset}-"
                f"n{args.n_parties}-"
                f"sr{args.sample}-"
                f"{args.train}-"
                f"lr{args.lr}-"
                f"hlr{args.head_lr}-"
                f"wd{args.weight_decay}-"
                f"{f'h{args.h}-' if args.train == 'fwd' else ''}"
                f"{f'fwdhead-' if args.fwd_head_tuning else ''}"
                f"{f'plt-' if args.partial_layer_training else ''}"
                f"{f'mr{args.mask_ratio}-' if args.train == 'fwd' else ''}"
                f"{f'mt{args.momentum_type}-' if args.momentum_type != 'none' and args.momentum != 0 else ''}"
                f"{f'mom{args.momentum}-' if args.momentum != 0 else ''}"
                f"{f'damp{args.dampening}-' if args.dampening != 0 else ''}"
                f"{f'clip{args.clip_norm}-' if args.clip_norm > 0 else ''}"
                f"ep{args.epochs}-"
                f"bs{args.batch_size}-"
                f"{f'K{args.batch_num}-' if args.batch_num != 0 else ''}"
                f"{f'N{args.perturb_num}-mbs{args.micro_batch_size}-nmb{args.num_micro_batch}-' if args.train == 'fwd' else ''}"
                f"{f'sl{args.sample_selection}-' if args.sample_selection != 'all' else ''}"
                f"{args.peftmode if args.peftmode != 'none' else ''}"
                f"{f'-r{args.r}' if args.peftmode == 'lora' else ''}"
                f"{f'-d{args.adapter_dim}' if args.peftmode == 'adapter' else ''}")

    root_path = args.logdir
    save_path = Path(os.path.join(root_path, save_path))
    save_path.mkdir(parents=True, exist_ok=True)

    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = args.device
    print(save_path)

    with open(os.path.join(save_path, 'commandline_args.txt'), 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset == 'tinyimagenet':
        NUM_CLASSES = 200
    elif args.dataset == 'cifar100':
        NUM_CLASSES = 100
    elif args.dataset == 'cifar10':
        NUM_CLASSES = 10
    elif 'sst2' in args.dataset:
        NUM_CLASSES = 2
    elif 'ag_news' in args.dataset:
        NUM_CLASSES = 4
    elif 'superglue' in args.dataset:
        if 'cb' in args.dataset:
            NUM_CLASSES = 3
        else:
            NUM_CLASSES = 2
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    data_loader_dict = {}
    train_ds, test_ds, net_dataidx_map_train, net_dataidx_map_test, _, _ = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir, args=args)

    for client_id in pit(range(args.n_parties)):
        data_loader_dict[client_id] = {}
        train_subset = Subset(train_ds, net_dataidx_map_train[client_id])
        test_subset = Subset(test_ds, net_dataidx_map_test[client_id])
        train_dl_local = DataLoader(dataset=train_subset, batch_size=min(args.batch_size, len(train_subset)), shuffle=True, drop_last=True)
        test_dl_local = DataLoader(dataset=test_subset, batch_size=64, shuffle=False, drop_last=False)
        data_loader_dict[client_id]['train_dl_local'] = train_dl_local
        data_loader_dict[client_id]['test_dl_local'] = test_dl_local

    ####### Model related ######
    if args.dataset in ['tinyimagenet', 'cifar100', 'cifar10']:
        from models.vit_models_fwd import VisionTransformerForward, CONFIGS
        config = CONFIGS['ViT-B_16']
        net = VisionTransformerForward(config, img_size=224, num_classes=NUM_CLASSES, vis=True, args=args)
        net.load_from(np.load('checkpoints/imagenet21k_ViT-B_16.npz'))
        net.freeze()
    else:
        from models.roberta_models import RobertaPEFT
        net = RobertaPEFT(num_classes=NUM_CLASSES, args=args)
    net.to(device)
    net = torch.compile(net, backend='inductor')

    global_para = {k: v.data.clone() for k, v in net.named_parameters() if v.requires_grad}
    clients_para = {}
    fed_avg_freqs = [len(data_loader_dict[r]['train_dl_local']) for r in range(args.n_parties)]
    client_first_grad = None

    if args.momentum_type in {'direct', 'replay-ht'}:
        peft_named_params, head_named_params = split_head_parameters(global_para)
        parameter_order = [*peft_named_params, *head_named_params]
        if not head_named_params:
            raise RuntimeError("No classification-head parameters were found in the trainable state")
    else:
        # Baseline modes retain their original parameter grouping.
        peft_named_params = {k:v for k,v in global_para.items() if 'classifier' not in k}
        head_named_params = {k:v for k,v in global_para.items() if 'classifier' in k}

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW([
            {'params': peft_named_params.values(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': head_named_params.values(), 'lr': args.head_lr, 'weight_decay': args.weight_decay}
        ])
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD([
            {'params': peft_named_params.values(), 'lr': args.lr, 'momentum': args.momentum, 'dampening': args.dampening, 'weight_decay': args.weight_decay},
            {'params': head_named_params.values(), 'lr': args.head_lr, 'momentum': args.momentum, 'dampening': args.dampening, 'weight_decay': args.weight_decay}
        ])
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    optim_state = optimizer.state_dict()

    if args.momentum_type == 'replay-ht':
        local_step_counts = sorted(
            {args.batch_num} if args.batch_num > 0 else set(fed_avg_freqs)
        )
        payload_ledger = {
            "protocol": "Exact HT/PEFT replay",
            "formula": "2S + 2H + 8T + 4JT",
            "by_local_steps": {
                str(local_steps): replay_ht_payload_bytes(
                    global_para,
                    head_named_params,
                    local_steps=local_steps,
                    jvps_per_step=args.perturb_num * args.num_micro_batch,
                )
                for local_steps in local_step_counts
            },
        }
        payload_label = "Exact HT/PEFT replay"
    elif args.momentum_type == 'direct':
        payload_ledger = direct_payload_bytes(global_para)
        payload_label = "C2Fed-Direct 4S"
    else:
        payload_ledger = None

    if payload_ledger is not None:
        with open(os.path.join(save_path, 'protocol_payload.json'), 'w', encoding='utf-8') as f:
            json.dump(payload_ledger, f, indent=2)
        if args.momentum_type == 'replay-ht':
            print(f"{payload_label} logical per-client payload: {json.dumps(payload_ledger['by_local_steps'], sort_keys=True)}")
        else:
            print(
                f"{payload_label} logical per-client payload: "
                f"down={payload_ledger['downlink']} bytes, "
                f"up={payload_ledger['uplink']} bytes, "
                f"total={payload_ledger['bidirectional']} bytes"
            )

    clients = np.arange(args.n_parties)
    num_sample = int(args.sample * args.n_parties)

    # Start federated learning
    best_acc = 0
    best_round = args.test_round

    from datetime import datetime
    csv_filename = os.path.join(save_path, 'results-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.csv')
    csv_file = open(csv_filename, 'w')
    csv_file.write('round,time,test_avg_loss,test_avg_acc,local_mean_acc,local_min_acc\n')
    csv_file.close()

    replay_diagnostics_filename = os.path.join(save_path, 'replay_diagnostics.jsonl')
    replay_timing_filename = os.path.join(save_path, 'replay_timing.csv')
    if args.momentum_type == 'replay-ht':
        with open(replay_timing_filename, 'w', encoding='utf-8') as timing_file:
            timing_file.write(
                'round,pre_aggregation_seconds,local_training_seconds,'
                'server_replay_seconds,audit_seconds\n'
            )

    if args.log_grad_norm:
        grad_nrom_csv_filename = os.path.join(save_path, 'grad_norm-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.csv')
        csv_file = open(grad_nrom_csv_filename, 'w')
        csv_file.write(f'round,{",".join(global_para.keys())}\n')
        csv_file.close()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(save_path)

    try:
        for comm_round in pit(range(args.comm_round), color='white'):
            start = time.time()
            server_replay_time = 0.0
            local_training_time = 0.0
            audit_time = 0.0
            round_replay_diagnostics = []

            selected_clients = np.random.choice(clients, num_sample, replace=False)
            param_dict = {}
            param_dict['round'] = comm_round
            if args.momentum_type in {'direct', 'replay-ht'}:
                param_dict['trainable_layer_names'] = set(global_para)
            else:
                # Baseline modes use their original trainable-layer selection.
                param_dict['trainable_layer_names'] = peft_named_params
                param_dict['trainable_layer_names'].update(head_named_params)
            client_optim_state = {}

            for client_id in pit(selected_clients, color='red'):
                net.load_state_dict(global_para, strict=False)
                train_dl_local = data_loader_dict[client_id]['train_dl_local']

                param_dict['train_dataloader'] = train_dl_local
                param_dict['optimizer_state'] = {'state':{}}
                for i, v in optim_state['state'].items():
                    param_dict['optimizer_state']['state'][i] = {'momentum_buffer': v['momentum_buffer'].data.clone()}

                if args.momentum_type == 'replay-ht':
                    local_steps = args.batch_num if args.batch_num > 0 else len(train_dl_local)
                    param_dict['replay_seeds'] = [
                        derive_replay_seed(args.init_seed, comm_round, int(client_id), step)
                        for step in range(local_steps)
                    ]
                    initial_peft_state = {
                        name: value.detach().clone()
                        for name, value in peft_named_params.items()
                    }
                    initial_named_momentum = named_momentum_from_optimizer_state(
                        optim_state,
                        parameter_order,
                        selected_names=set(peft_named_params),
                    )
                    initial_peft_momentum = {
                        name: initial_named_momentum[name]
                        for name in peft_named_params
                        if name in initial_named_momentum
                    }
                else:
                    param_dict.pop('replay_seeds', None)

                if args.momentum_type == 'replay-ht':
                    local_training_start = time.time()
                if args.train == 'fwd':
                    net, returned_optim_state = train_local_fwd(net, args, param_dict, client_first_grad)
                else:
                    net, returned_optim_state = train_local_bp(net, args, param_dict)
                if args.momentum_type == 'replay-ht':
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    local_training_time += time.time() - local_training_start

                if args.momentum_type == 'replay-ht' and args.replay_performance_mode:
                    client_head_state = {
                        name: value.data.clone()
                        for name, value in net.named_parameters()
                        if value.requires_grad and name in head_named_params
                    }
                else:
                    client_full_state = {k: v.data.clone() for k, v in net.named_parameters() if v.requires_grad}

                if args.momentum_type == 'replay-ht':
                    trace = param_dict.pop('replay_trace_output')
                    expected_jvps = args.perturb_num * args.num_micro_batch
                    if len(trace) != local_steps or any(
                        step_trace['jvps'].numel() != expected_jvps
                        for step_trace in trace
                    ):
                        raise RuntimeError(
                            f"invalid replay trace for client {client_id}: "
                            f"steps={len(trace)}, expected_steps={local_steps}"
                        )
                    replay_start = time.time()
                    replayed_peft_state, replayed_peft_momentum = replay_peft_trace(
                        initial_peft_state,
                        initial_peft_momentum,
                        trace,
                        learning_rate=args.lr,
                        momentum=args.momentum,
                        dampening=args.dampening,
                        weight_decay=args.weight_decay,
                        smooth_factor=args.h,
                        mask_ratio=args.mask_ratio,
                        clip_norm=args.clip_norm,
                    )
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    server_replay_time += time.time() - replay_start

                    audit_start = time.time()
                    if args.replay_performance_mode:
                        client_named_momentum = named_momentum_from_optimizer_state(
                            returned_optim_state,
                            parameter_order,
                            selected_names=set(head_named_params),
                        )
                    else:
                        client_named_momentum = named_momentum_from_optimizer_state(
                            returned_optim_state,
                            parameter_order,
                        )
                        parameter_max_abs = max(
                            (
                                replayed_peft_state[name] - client_full_state[name]
                            ).abs().max().item()
                            for name in peft_named_params
                        )
                        momentum_max_abs = max(
                            (
                                replayed_peft_momentum[name] - client_named_momentum[name]
                            ).abs().max().item()
                            for name in peft_named_params
                        )
                        for name in peft_named_params:
                            torch.testing.assert_close(
                                replayed_peft_state[name],
                                client_full_state[name],
                                rtol=0,
                                atol=1e-6,
                            )
                            torch.testing.assert_close(
                                replayed_peft_momentum[name],
                                client_named_momentum[name],
                                rtol=0,
                                atol=1e-6,
                            )
                    audit_time += time.time() - audit_start

                    clients_para[client_id] = {
                        name: (
                            replayed_peft_state[name]
                            if name in replayed_peft_state
                            else (
                                client_head_state[name]
                                if args.replay_performance_mode
                                else client_full_state[name]
                            )
                        )
                        for name in global_para
                    }
                    combined_momentum = {
                        **replayed_peft_momentum,
                        **{
                            name: client_named_momentum[name]
                            for name in head_named_params
                        },
                    }
                    client_optim_state[client_id] = optimizer_state_from_named_momentum(
                        combined_momentum,
                        parameter_order,
                        optimizer.state_dict(),
                    )
                    client_payload = replay_ht_payload_bytes(
                        global_para,
                        head_named_params,
                        local_steps=local_steps,
                        jvps_per_step=expected_jvps,
                    )
                    if not args.replay_performance_mode:
                        round_replay_diagnostics.append({
                            "round": comm_round,
                            "client_id": int(client_id),
                            "local_steps": local_steps,
                            "jvp_count": sum(step['jvps'].numel() for step in trace),
                            "parameter_max_abs_error": parameter_max_abs,
                            "momentum_max_abs_error": momentum_max_abs,
                            "payload": client_payload,
                        })
                else:
                    clients_para[client_id] = client_full_state
                if args.momentum_type not in {'direct', 'replay-ht'}:
                    client_optim_state[client_id] = returned_optim_state

            if args.momentum_type == 'replay-ht' and str(device).startswith('cuda'):
                torch.cuda.synchronize()
            client_train_time = time.time() - start

            if args.momentum_type == 'replay-ht' and round_replay_diagnostics:
                with open(replay_diagnostics_filename, 'a', encoding='utf-8') as replay_file:
                    for record in round_replay_diagnostics:
                        replay_file.write(json.dumps(record, sort_keys=True) + '\n')
            if args.momentum_type == 'replay-ht':
                with open(replay_timing_filename, 'a', encoding='utf-8') as timing_file:
                    timing_file.write(
                        f'{comm_round},{client_train_time:.6f},'
                        f'{local_training_time:.6f},{server_replay_time:.6f},'
                        f'{audit_time:.6f}\n'
                    )

            optimizer.zero_grad()
            global_para = fedavg(
                clients_para,
                global_para,
                fed_avg_freqs,
                selected_clients,
                update_para=(args.momentum_type != 'server'),
            )

            if args.log_grad_norm:
                csv_file = open(grad_nrom_csv_filename, 'a')
                csv_file.write(f'{comm_round},{",".join([f"{v.grad.norm():f}" for v in global_para.values()])}\n')
                csv_file.close()

            if args.momentum_type in {'direct', 'replay-ht'}:
                optim_state['state'] = {i:{"momentum_buffer":torch.stack([client_optim_state[client_id]['state'][i]['momentum_buffer'] for client_id in selected_clients]).mean(dim=0)} for i, p in enumerate(global_para.values())}
            elif args.momentum_type == 'fedavg-m':
                scale = 1 / (args.lr * np.mean([fed_avg_freqs[client_id] for client_id in selected_clients]))
                optim_state['state'] = {i:{'momentum_buffer': p.grad*scale} for i, p in enumerate(global_para.values())}
            elif args.momentum_type == 'server':
                for i, p in enumerate(global_para.values()):
                    p.grad /= args.lr
                optimizer.step()
                optim_state = {'state':{}}

            clients_para = {}
            client_optim_state = {}

            if comm_round % args.test_freq == 0 or comm_round >= args.test_round:
                print(f'########### Now is the round {comm_round} ######')
                print(f"Client per round training time = {client_train_time:.2f}s")
                if args.momentum_type == 'replay-ht':
                    print(f"Server replay time = {server_replay_time:.2f}s")
                    print(f"Local training time = {local_training_time:.2f}s")
                    print(f"Replay audit time = {audit_time:.2f}s")
                test_results, test_avg_loss, test_avg_acc, local_mean_acc, local_min_acc = compute_accuracy(net, global_para, data_loader_dict, args)
                print(f'>> Mean Local Test accuracy: {local_mean_acc:.3f}')
                print(f'>> Min Local Test accuracy: {local_min_acc:.3f}')
                print(f'>> Global Model Test accuracy: {test_avg_acc:.3f}')
                print(f'>> Test avg loss: {test_avg_loss:.3f}')

                csv_file = open(csv_filename, 'a')
                csv_file.write(f'{comm_round},{client_train_time:.2f},{test_avg_loss:.3f},{test_avg_acc:.3f},{local_mean_acc:.3f},{local_min_acc:.3f}\n')
                csv_file.close()

                writer.add_scalar('test_avg_loss', test_avg_loss, comm_round)
                writer.add_scalar('test_avg_acc', test_avg_acc, comm_round)
                writer.add_scalar('local_mean_acc', local_mean_acc, comm_round)
                writer.add_scalar('local_min_acc', local_min_acc, comm_round)

                if test_avg_acc >= best_acc:
                    torch.save(global_para, os.path.join(save_path, 'Roberta_PEFT.pt'))
                    torch.save(optim_state, os.path.join(save_path, 'optimizer_state.pt'))
                    best_acc = test_avg_acc
                    best_round = comm_round
                    writer.add_scalar('best_acc', best_acc, comm_round)
                elif comm_round - best_round >= args.patience:
                    break
    finally:
        writer.add_hparams(args.__dict__, {'best_acc': best_acc})
        writer.close()

if __name__ == "__main__":
    main()
