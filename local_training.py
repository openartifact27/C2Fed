"""Local training kernels used by the reported C2Fed experiments."""

import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from progress import pit


def generate_random_masks(elements, num_masks, mask_size):
    all_masks = []
    element_counts = {elem: 0 for elem in elements}

    for _ in range(num_masks):
        elements = np.random.permutation(elements)
        sorted_elements = sorted(elements, key=lambda x: element_counts[x])
        mask = sorted_elements[:mask_size]
        all_masks.append(mask)
        for elem in mask:
            element_counts[elem] += 1

    return all_masks, element_counts


def masks_to_one_hot(masks, num_elements):
    one_hot_masks = []
    for mask in masks:
        one_hot = [1 if i in mask else 0 for i in range(num_elements)]
        one_hot_masks.append(one_hot)
    return one_hot_masks


@torch.no_grad()
@torch.compile(backend='inductor')
def forward_head_tuning(y, labels):
    batch_size = y['logits'].shape[0]
    softmax_y = F.softmax(y['logits'], dim=-1)
    softmax_y[torch.arange(batch_size),labels] -= 1
    grad = softmax_y.div_(batch_size)
    return torch.sum(grad, dim=0), grad.T @ y['cls_token']


@torch.no_grad()
@torch.compile(backend='inductor')
def perturb(fwd_para, fwd_para_data, mask, h):
    p_params = []
    for p, d, m in zip(fwd_para, fwd_para_data, mask):
        perturb = torch.randn_like(d).mul_(m).mul_(h)
        p_params.append(perturb)
        p.data = d + perturb
    return p_params


@torch.no_grad()
def perturb_seeded(fwd_para, fwd_para_data, mask, h, generator):
    """Apply one replayable perturbation using a protocol-owned RNG."""

    p_params = []
    for p, d, m in zip(fwd_para, fwd_para_data, mask):
        noise = torch.randn(
            d.shape,
            dtype=d.dtype,
            device=d.device,
            generator=generator,
        ).mul_(m).mul_(h)
        p_params.append(noise)
        p.data = d + noise
    return p_params


@torch.no_grad()
def fwd_perturb(
    net,
    data,
    label,
    losses,
    idx,
    fwd_para,
    fwd_para_data,
    mask,
    h,
    generator=None,
):
    if generator is None:
        p_params = perturb(fwd_para, fwd_para_data, mask, h)
    else:
        p_params = perturb_seeded(
            fwd_para,
            fwd_para_data,
            mask,
            h,
            generator,
        )
    loss = losses[idx].mean()
    if isinstance(data, dict):
        x = data['input_ids'][idx], data['attention_mask'][idx]
    else:
        x = data[idx]
    jvp = F.cross_entropy(net(x)['logits'], label[idx]).sub_(loss).div_(h)
    return jvp, p_params


@torch.no_grad()
def fwd_train_step(
    net,
    x,
    labels,
    fwd_para,
    fwd_para_data,
    args,
    replay_seed=None,
):
    if isinstance(x, dict):
        y = net(x['input_ids'], x['attention_mask'])
    else:
        y = net(x)
    losses = F.cross_entropy(y['logits'], labels, reduction='none')
    if args.sample_selection == 'uncertainty':
        filter_idxs = torch.argsort(losses)[-args.micro_batch_size * (args.num_micro_batch):]
    elif args.sample_selection == 'random':
        filter_idxs = torch.arange(args.micro_batch_size * args.num_micro_batch)
    elif args.sample_selection == 'all':
        filter_idxs = torch.arange(len(losses))
    if args.fwd_head_tuning:
        net.head.bias.grad, net.head.weight.grad = forward_head_tuning(y, labels)
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.head.parameters(), args.clip_norm)
    num_micro_batch = len(filter_idxs) // args.micro_batch_size
    filter_idxs_list = np.array_split(filter_idxs[torch.randperm(len(filter_idxs))], num_micro_batch)
    replay_generator = None
    if replay_seed is None:
        mask = [torch.rand_like(v) > args.mask_ratio for v in fwd_para_data]
    else:
        generator_device = fwd_para_data[0].device
        replay_generator = torch.Generator(device=generator_device)
        replay_generator.manual_seed(replay_seed)
        mask = [
            torch.rand(
                v.shape,
                dtype=v.dtype,
                device=v.device,
                generator=replay_generator,
            ) > args.mask_ratio
            for v in fwd_para_data
        ]
    replay_jvps = []
    for idx in filter_idxs_list:
        for i in range(args.perturb_num):
            jvp, p_params = fwd_perturb(
                net,
                x,
                labels,
                losses,
                idx,
                fwd_para,
                fwd_para_data,
                mask,
                args.h,
                generator=replay_generator,
            )
            if replay_seed is not None:
                # Keep scalars device-resident throughout the step and expose
                # one contiguous tensor.  Calling ``to(cpu)`` here forced a
                # device synchronization for every directional derivative.
                replay_jvps.append(jvp.detach().to(dtype=torch.float32))
            for p, perturbation in zip(fwd_para, p_params):
                # Evaluated convention: perturbation=h*v and
                # jvp=(L(theta+h*v)-L(theta))/h, so the accumulated update is
                # (L(theta+h*v)-L(theta))*v before group-wise clipping.
                p.grad.add_(perturbation.mul_(jvp))
    for p, d in zip(fwd_para, fwd_para_data):
        p.data = d
    if args.clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(fwd_para, args.clip_norm)
    if replay_seed is None:
        return None
    return {
        "seed": int(replay_seed),
        "jvps": torch.stack(replay_jvps),
    }


def train_local_fwd(net, args, param_dict, client_first_grad=None):
    train_dataloader = param_dict['train_dataloader']
    all_trainable_named_para = {k:v for k,v in net.named_parameters() if v.requires_grad}
    all_trainable_para = all_trainable_named_para.values()
    peft_named_params = {k:v for k,v in all_trainable_named_para.items() if 'head' not in k}
    head_named_params = {k:v for k,v in all_trainable_named_para.items() if 'head' in k}
    if args.fwd_head_tuning:
        fwd_named_para = peft_named_params
    else:
        fwd_named_para = all_trainable_named_para
    fwd_named_para = {k:v for k,v in fwd_named_para.items() if k in param_dict['trainable_layer_names']}
    fwd_para = fwd_named_para.values()
    fwd_para_data = [v.data for v in fwd_para]
    for p in all_trainable_para:
        p.grad = torch.zeros_like(p.data)
        p.requires_grad = False
    net.eval()
    param_group = [{'params':peft_named_params.values(),'lr':args.lr,'momentum':args.momentum, 'dampening':args.dampening, 'weight_decay':args.weight_decay},
                    {'params':head_named_params.values(),'lr':args.head_lr,'momentum':args.momentum, 'dampening':args.dampening, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.SGD(param_group)
    if param_dict['optimizer_state'] is not None:
        optim_state = optimizer.state_dict()
        optim_state.update(param_dict['optimizer_state'])
        optimizer.load_state_dict(copy.deepcopy(optim_state))
    if args.batch_num != 0:
        batch_num = args.batch_num
    else:
        batch_num = len(train_dataloader)
    batch_pit = pit(range(batch_num), color = 'blue') if batch_num > 1 else range(batch_num)
    train_iter = itertools.cycle(train_dataloader)
    replay_seeds = param_dict.get('replay_seeds')
    if replay_seeds is not None and len(replay_seeds) != batch_num:
        raise ValueError(
            f"expected {batch_num} replay seeds, received {len(replay_seeds)}"
        )
    replay_trace = []
    with torch.no_grad():
        for batch_idx in batch_pit:
            batch = next(train_iter)
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['label'].to(args.device)
                x = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                images, labels = batch
                images, labels = images.to(args.device), labels.long().to(args.device)
                x = images
            replay_seed = None if replay_seeds is None else replay_seeds[batch_idx]
            step_trace = fwd_train_step(
                net,
                x,
                labels,
                fwd_para,
                fwd_para_data,
                args,
                replay_seed=replay_seed,
            )
            if step_trace is not None:
                replay_trace.append(step_trace)
            if args.momentum_type == 'fedavg-m':
                optimizer.load_state_dict(copy.deepcopy(optim_state))
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)

    for p in all_trainable_para:
        p.requires_grad = True
    if replay_seeds is not None:
        param_dict['replay_trace_output'] = replay_trace
    return net, optimizer.state_dict()


def train_local_bp(net, args, param_dict):
    train_dataloader = param_dict['train_dataloader']
    trainable_named_para = {k:v for k,v in net.named_parameters() if v.requires_grad}
    peft_named_params = {k:v for k,v in trainable_named_para.items() if 'head' not in k}
    head_named_params = {k:v for k,v in trainable_named_para.items() if 'head' in k}
    optimizer = optim.SGD([{'params':peft_named_params.values(),'lr':args.lr,'momentum':args.momentum, 'dampening':args.dampening, 'weight_decay':args.weight_decay},
                           {'params':head_named_params.values(),'lr':args.head_lr,'momentum':args.momentum, 'dampening':args.dampening, 'weight_decay':args.weight_decay}])
    if param_dict['optimizer_state'] is not None:
        optim_state = optimizer.state_dict()
        optim_state.update(param_dict['optimizer_state'])
        optimizer.load_state_dict(copy.deepcopy(optim_state))
    criterion = torch.nn.CrossEntropyLoss()
    if args.batch_num != 0:
        batch_num = args.batch_num
    else:
        batch_num = len(train_dataloader)
    batch_pit = pit(range(batch_num), color = 'blue') if batch_num > 1 else range(batch_num)
    train_iter = itertools.cycle(train_dataloader)
    for batch_idx in batch_pit:
        batch = next(train_iter)
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)
            outputs = net(input_ids, attention_mask)
        else:
            images, labels = batch
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)

        logits = outputs['logits']
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.momentum_type == 'fedavg-m':
            optimizer.load_state_dict(copy.deepcopy(optim_state))

    return net, optimizer.state_dict()
