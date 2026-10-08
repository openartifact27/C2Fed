import argparse
import copy
import json
import os
import random
import time
import math
from collections import defaultdict
from pathlib import Path
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.roberta_models import RobertaPEFT
from utils import (
    partition_data,
    compute_accuracy,
    fedavg,
    pit
)

GRAD_CHECK_LAYER_NAME = '_orig_mod.head.weight'

@torch.no_grad()
@torch.compile()
def calculate_cos_sim(A,target_grad):
    batch_size = 1000

    # 计算总批次数
    num_batches = math.ceil(A.size(0) / batch_size)

    # 创建一个空的结果张量
    result = torch.empty(A.size(0))

    # 逐批次计算余弦相似度
    for i in range(num_batches):
        # 获取当前批次的起始索引和结束索引
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, A.size(0))

        # 提取当前批次的向量
        batch = A[start_idx:end_idx]

        # 计算当前批次的余弦相似度
        similarity = torch.cosine_similarity(batch, target_grad, dim=-1)

        # 将结果保存到结果张量的对应位置
        result[start_idx:end_idx] = similarity

    return result

@torch.no_grad()
@torch.compile(backend='inductor')
def computevar(fwdgrad_list):
    fwdgrad_list = torch.cat(fwdgrad_list)
    n = fwdgrad_list.shape[0]
    # 计算前一半tensor的平均值
    first_half_mean = torch.mean(fwdgrad_list[:n//2], dim=0)

    # 计算后一半tensor的平均值
    second_half_mean = torch.mean(fwdgrad_list[n//2:], dim=0)

    # 计算两个平均值之间的方差
    # 这里是算的两个half_mean tensor的首个维度对应元素的方差，返回结果有多少个元素就有多少个方差，然后对这些方差求个mean，返回这个mean
    var = torch.var(torch.stack([first_half_mean, second_half_mean]), dim=0).mean()

    return var

@torch.no_grad()
@torch.compile(backend='inductor')
def sample_perturb(client_first_grad, batch_num, v_num):
    v_buffer = {}
    for k, v in client_first_grad.items():
        target_grad = torch.flatten(v)
        candidate_v = torch.randn((batch_num*v_num, v.numel()),device="cuda")
        cos_sim = calculate_cos_sim(candidate_v,target_grad)
        _, sorted_indices = torch.sort(cos_sim, descending=True)
        v_buffer[k] = candidate_v[sorted_indices[:batch_num]]
    return v_buffer

@torch.no_grad()
@torch.compile(backend='inductor')
def perturb_para(fwd_para, fwd_para_data, v_buffer, idx, h):
    for perturb, p, d in zip(v_buffer.values(), fwd_para, fwd_para_data):
        p.data = d + perturb[idx].reshape(d.shape).mul(h)

@torch.no_grad()
@torch.compile(backend='inductor')
def restore_para(fwd_para, fwd_para_data):
    for p, d in zip(fwd_para, fwd_para_data):
        p.data = d

@torch.no_grad()
def train_local_fwdllm(net, args, param_dict, client_first_grad={}):
    train_dataloader = param_dict['train_dataloader']
    all_trainable_named_para = {k:v for k,v in net.named_parameters() if v.requires_grad}
    # print(all_trainable_named_para.keys())
    all_trainable_para = all_trainable_named_para.values()
    fwd_named_para = all_trainable_named_para
    fwd_para = fwd_named_para.values()
    fwd_para_data = [v.data for v in fwd_para]
    for p in all_trainable_para:
        p.grad = torch.zeros_like(p.data)
        p.requires_grad = False
    if args.batch_num > 0:
        batch_num = args.batch_num
    else:
        batch_num = len(train_dataloader)
    if client_first_grad:
        v_buffer = sample_perturb(client_first_grad, batch_num * args.perturb_num, args.v_num)
    else:
        v_buffer = {k: torch.randn((batch_num * args.perturb_num, v.numel()),device="cuda") for k, v in all_trainable_named_para.items()}

    jvp_list = []
    idx = 0
    train_iter = itertools.cycle(train_dataloader)
    for batch_idx in pit(range(batch_num), color='blue'):
        batch = next(train_iter)
        if isinstance(batch, dict):
            x = batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)
        else:
            x, labels = batch[0].to(args.device), batch[1].to(args.device)
        for _ in range(args.perturb_num): # number of perturbations for each batch
            perturb_para(fwd_para, fwd_para_data, v_buffer, idx, args.h)
            y_pos = net(x)
            loss_pos = F.cross_entropy(y_pos['logits'], labels)
            perturb_para(fwd_para, fwd_para_data, v_buffer, idx, -args.h)
            y_neg = net(x)
            loss_neg = F.cross_entropy(y_neg['logits'], labels)
            jvp = (loss_pos - loss_neg).div_(2*args.h)
            jvp_list.append(jvp)
            idx += 1
    jvp = torch.stack(jvp_list).unsqueeze(1)
    for (k, perturb), p, d in zip(v_buffer.items(), fwd_para, fwd_para_data):
        perturb.mul_(jvp)
        if k == GRAD_CHECK_LAYER_NAME:
            grad_check_list = perturb
        p.data = d - perturb.mean(dim=0).reshape(p.shape).mul_(args.lr)

    for p in all_trainable_para:
        p.requires_grad = True
    return net, grad_check_list

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='roberta-base', help='model name')
    parser.add_argument('--dataset', type=str, default='sst2', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-labeluni', help='the data partitioning strategy')
    parser.add_argument('--cls_num', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-2)')
    parser.add_argument('--head_lr', type=float, default=0, help='learning rate (default: lr)')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=60, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for optimizier')
    parser.add_argument('--dampening', type=float, default=0, help='dampening for optimizier')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay for optimizier')
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--test_round', type=int, default=50, help='The round of testing')
    parser.add_argument('--test_freq', type=int, default=5, help='The frequency of testing')
    parser.add_argument('--patience', type=int, default=100, help='The patience of early stopping')
    parser.add_argument('--root_path', type=str, default='', help='Noise type: None/increasng/space')
    parser.add_argument('--check_layer_id', type=int, default=49)
    parser.add_argument('--var_threshold', type=float, default=0.1)
    parser.add_argument('--v_num', type=int, default=10)
    parser.add_argument('--h', type=float, default=0.01, help='perturbation rate (default: 0.01)')
    parser.add_argument('-N', '--perturb_num', type=int, default=10, help='num of noise in fwd perturb')
    parser.add_argument('-K', '--batch_num', type=int, default=0, help='num of noise in fwd perturb')
    parser.add_argument('--train',type=str,default='fwdllm',choices=['fwdllm'])
    parser.add_argument('--peftmode', type=str, default='full-ft', choices=['head', 'adapter', 'lora', 'bitfit', 'full-ft'], help='set peft mode')
    parser.add_argument('--adapter_dim', type=int, default=8, help='The dimension of the adapter')
    parser.add_argument('--alpha', type=int, default=8, help='The alpha of the lora')
    parser.add_argument('--r', type=int, default=8, help='The r of the lora')
    parser.add_argument('--fwd_head_tuning',action='store_true')
    parser.add_argument('--partial_layer_training',action='store_true')
    parser.add_argument('--sample_selection', type=str, default='all', choices=['all', 'uncertainty', 'random'], help='sample selection method')
    parser.add_argument('--momentum_type', choices=['server', 'client', 'fedavg-m', 'none'], default='none', help='Whether to use client optimizer state')
    parser.add_argument('--log_grad_norm', action='store_true', help='Whether to log the gradient norm')
    return parser.parse_args()

def main():
    args = get_args()

    if args.head_lr == 0:
        args.head_lr = args.lr

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
                f"bs{args.batch_size}-"
                f"{f'K{args.batch_num}-' if args.batch_num != 0 else ''}"
                f"N{args.perturb_num}-"
                f"{f'sl{args.sample_selection}-' if args.sample_selection != 'all' else ''}"
                f"{args.peftmode if args.peftmode != 'none' else ''}"
                f"{f'-r{args.r}' if args.peftmode == 'lora' else ''}"
                f"{f'-d{args.adapter_dim}' if args.peftmode == 'adapter' else ''}")
    root_path = args.logdir

    save_path = Path(os.path.join(root_path,save_path))
    save_path.mkdir(parents=True, exist_ok=True)
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = args.device
    print(save_path)

    with open(os.path.join(save_path,'commandline_args.txt'), 'w', encoding='utf-8') as f:
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
    train_ds, test_ds, net_dataidx_map_train, net_dataidx_map_test, _ , _= partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir,args=args)
    for client_id in pit(range(args.n_parties)):
        data_loader_dict[client_id] = {}
        train_subset = Subset(train_ds, net_dataidx_map_train[client_id])
        test_subset = Subset(test_ds, net_dataidx_map_test[client_id])
        train_dl_local = DataLoader(dataset=train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_dl_local = DataLoader(dataset=test_subset, batch_size=64, shuffle=False, drop_last=False)
        data_loader_dict[client_id]['train_dl_local'] = train_dl_local
        data_loader_dict[client_id]['test_dl_local'] = test_dl_local

    ###### Model related ######
    if args.dataset in ['tinyimagenet', 'cifar100', 'cifar10']:
        from models.vit_models_fwd import VisionTransformerForward, CONFIGS
        config = CONFIGS['ViT-B_16']
        net = VisionTransformerForward(config, img_size=224, num_classes=NUM_CLASSES, vis=True, args=args)
        net.load_from(np.load('checkpoints/imagenet21k_ViT-B_16.npz'))
        net.freeze()
    else:
        net = RobertaPEFT(num_classes=NUM_CLASSES, args=args)
    net.to(device)
    net = torch.compile(net, backend='inductor')

    global_para = {k: v.data.clone() for k, v in net.named_parameters() if v.requires_grad}
    clients_para = {}

    global_grad = {}
    fed_avg_freqs = [len(data_loader_dict[r]['train_dl_local']) for r in range(args.n_parties)]
    clients = np.arange(args.n_parties)
    num_sample = int(args.sample * args.n_parties)
    # 开始fed_learning
    best_acc = 0
    best_round = args.test_round
    from datetime import datetime
    csv_filename = os.path.join(save_path,'results-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.csv')
    csv_file = open(csv_filename, 'w')
    csv_file.write('round,time,var,test_avg_loss,test_avg_acc,local_mean_acc,local_min_acc\n')
    csv_file.close()

    if args.log_grad_norm:
        grad_nrom_csv_filename = os.path.join(save_path,'grad_norm-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.csv')
        csv_file = open(grad_nrom_csv_filename, 'w')
        csv_file.write(f'round,{",".join(global_para.keys())}\n')
        csv_file.close()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(save_path)
    try:
        for comm_round in range(args.comm_round):
            print(f'########### Now is the round {comm_round} ######')
            start = time.time()

            # 维护每轮的前向梯度
            selected_clients = np.random.choice(clients, num_sample, replace=False)
            param_dict = {}
            param_dict['round'] = comm_round
            if comm_round % 2 != 0:
                global_grad = {}
            grad_check_list = []
            for client_id in pit(selected_clients, color = 'red'):
                net.load_state_dict(global_para, strict=False)
                param_dict['train_dataloader'] = data_loader_dict[client_id]['train_dl_local']
                net, _grad_check_list = train_local_fwdllm(net, args, param_dict, global_grad)
                clients_para[client_id] = {k: v.data.clone() for k, v in net.named_parameters() if v.requires_grad}
                grad_check_list.append(_grad_check_list)
            var = computevar(grad_check_list)
            print(f"Var: {var}")
            if var > args.var_threshold:
                retry_cnt = 1
                for client_id in selected_clients:
                    clients_para[client_id] = {k: v.unsqueeze(0) for k, v in clients_para[client_id].items()}
                while var > args.var_threshold:
                    print(f"Var is too large, train again")
                    for client_id in pit(selected_clients, color = 'red'):
                        net.load_state_dict(global_para, strict=False)
                        param_dict['train_dataloader'] = data_loader_dict[client_id]['train_dl_local']
                        net, _grad_check_list = train_local_fwdllm(net, args, param_dict, global_grad)
                        for k, v in net.named_parameters():
                            if v.requires_grad:
                                clients_para[client_id][k] = torch.cat([clients_para[client_id][k], v.data.unsqueeze(0)], dim=0)
                        grad_check_list.append(_grad_check_list)
                    var = computevar(grad_check_list)
                    print(f"Var: {var}")
                    retry_cnt += 1
                    if retry_cnt > 10:
                        print(f"Var is too large, retried too many times, break")
                        break
                for client_id in selected_clients:
                    clients_para[client_id] = {k: v.mean(dim=0) for k, v in clients_para[client_id].items()}
            client_train_time = time.time() - start
            print(f"Client training time = {client_train_time:.2f}s")
            global_para = fedavg(clients_para, global_para, fed_avg_freqs, selected_clients, update_para=(args.momentum_type != 'server'))
            global_grad = {k:v.grad for k,v in global_para.items()}
            if args.log_grad_norm:
                global_grad_norm = {k:v.grad.norm() for k,v in global_para.items()}
                csv_file = open(grad_nrom_csv_filename, 'a')
                csv_file.write(f'{comm_round},{",".join([f"{v.grad.norm():f}" for v in global_para.values()])}\n')
                csv_file.close()
            clients_para = {}

            if comm_round % args.test_freq == 0 or comm_round >= args.test_round:
                test_results, test_avg_loss, test_avg_acc, local_mean_acc,local_min_acc = compute_accuracy(net, global_para, data_loader_dict, args)
                print(f'>> Mean Local Test accuracy: {local_mean_acc:.3f}')
                print(f'>> Min Local Test accuracy: {local_min_acc:.3f}')
                print(f'>> Global Model Test accuracy: {test_avg_acc:.3f}')
                print(f'>> Test avg loss: {test_avg_loss:.3f}')
                csv_file = open(csv_filename, 'a')
                csv_file.write(f'{comm_round},{client_train_time:.2f},{var:.3f},{test_avg_loss:.3f},{test_avg_acc:.3f},{local_mean_acc:.3f},{local_min_acc:.3f}\n')
                csv_file.close()

                writer.add_scalar('test_avg_loss', test_avg_loss, comm_round)
                writer.add_scalar('test_avg_acc', test_avg_acc, comm_round)
                writer.add_scalar('local_mean_acc', local_mean_acc, comm_round)
                writer.add_scalar('local_min_acc', local_min_acc, comm_round)

                if test_avg_acc >= best_acc:
                    torch.save(global_para,os.path.join(save_path,'Roberta_PEFT.pt'))
                    best_acc = test_avg_acc
                    best_round = comm_round
                    writer.add_scalar('best_acc', best_acc, comm_round)
                elif comm_round - best_round >= args.patience:
                    break
    finally:
        writer.add_hparams(args.__dict__, {'best_acc': best_acc})
        writer.close()
if __name__ == '__main__':
    main()