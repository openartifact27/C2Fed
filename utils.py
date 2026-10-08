import os
import logging
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
import itertools

import numpy as np
import torchvision.transforms.v2 as transforms
import torchvision

import torch
import torch.nn as nn
from progress import pit

# from pypapi import events, papi_high as high

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

@torch.no_grad()
def compute_accuracy(net, global_para, data_loader_dict, args):
    test_results = defaultdict(lambda: defaultdict(list))
    net.load_state_dict(global_para, strict=False)
    net.eval()

    for net_id in range(args.n_parties):
        if net_id not in data_loader_dict.keys():
            continue
        test_dl_local = data_loader_dict[net_id]['test_dl_local']
        test_correct, test_total, test_avg_loss = eval_local_model(net, test_dl_local, device=args.device, args=args)
        test_results[net_id]['loss'] = test_avg_loss
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    ### local performance
    local_mean_acc = np.mean([val['correct']/val['total'] for val in test_results.values()])
    local_min_acc = np.min([val['correct']/val['total'] for val in test_results.values()])

    return test_results, test_avg_loss, test_avg_acc, local_mean_acc, local_min_acc

def eval_local_model(net, data_loader, device, args):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            if args.dataset == 'sst2' or args.dataset == 'ag_news' or 'superglue' in args.dataset:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = net(input_ids, attention_mask)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)

            logits = outputs['logits']
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total, total_loss / len(data_loader)

def fedavg(clients_para, global_para, fed_avg_freqs, selected_clients, update_para=True):
    for k,v in global_para.items():
        new_p = torch.stack([clients_para[client_id][k] for client_id in selected_clients if k in clients_para[client_id]]).mean(dim=0)
        v.grad = v.data - new_p
        if update_para:
            v.data = new_p
    return global_para

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar10_data(datadir):
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        torch.compile(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    ])
    train_ds = torchvision.datasets.CIFAR10(datadir, train=True, download=True, transform=transform, target_transform=lambda x: torch.tensor(x))
    test_ds = torchvision.datasets.CIFAR10(datadir, train=False, download=True, transform=transform, target_transform=lambda x: torch.tensor(x))
    return train_ds, test_ds

def load_cifar100_data(datadir):
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        torch.compile(transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)))
    ])

    train_ds = torchvision.datasets.CIFAR100(datadir, train=True, download=True, transform=transform, target_transform=lambda x: torch.tensor(x))
    test_ds = torchvision.datasets.CIFAR100(datadir, train=False, download=True, transform=transform, target_transform=lambda x: torch.tensor(x))

    return train_ds, test_ds

def load_tinyimagenet_data(datadir):
    from cifar100_datasets import NPZDataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32)
    ])
    train_ds = NPZDataset(os.path.join(datadir, 'tinyimagenet_train.npz'), transform=transform)
    test_ds = NPZDataset(os.path.join(datadir, 'tinyimagenet_val.npz'), transform=transform)

    return train_ds, test_ds

def SST2Template(sample):
    sentence = sample["sentence"].strip()
    return f"{sentence} It was"

def load_sst2_data(tokenizer):
    from datasets import load_dataset

    dataset = load_dataset('sst2')

    def tokenize_function(examples):
        batch_size = len(examples['label'])
        texts = [SST2Template({k: examples[k][i] for k in examples}) for i in range(batch_size)]
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

    # Tokenize datasets
    train_ds = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=['sentence']  # Only remove the sentence column
    )
    test_ds = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=['sentence']  # Only remove the sentence column
    )

    # Convert to PyTorch format
    train_ds.set_format('torch')
    test_ds.set_format('torch')

    return train_ds, test_ds

def BoolQTemplate(sample):
    passage = sample["passage"]
    question = sample["question"]
    return f"{passage}\nQuestion: {question}\nIs that correct? Yes or No?\n"

def RTETemplate(sample):
    premise = sample["premise"]
    hypothesis = sample["hypothesis"]
    return f'{premise}\nDoes this mean that "{hypothesis}" is true? Yes or No?\n'

def MultiRCTemplate(sample):
    paragraph = sample["paragraph"]
    question = sample["question"]
    answer = sample["answer"]
    return f'{paragraph}\nQuestion: {question}\nI found this answer "{answer}". Is that correct? Yes or No?\n'

def CBTemplate(sample):
    premise = sample["premise"]
    hypothesis = sample["hypothesis"]
    return f'Suppose {premise} Can we infer that "{hypothesis}"? Yes, No, or Maybe?\n'

def WICTemplate(sample):
    sent1 = sample["sentence1"]
    sent2 = sample["sentence2"]
    word = sample["word"]
    return f'Does the word "{word}" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n'

def WSCTemplate(sample):
    text = sample["text"]
    span1 = sample["span1_text"]
    span2 = sample["span2_text"]
    return f'{text}\nIn the previous sentence, does the pronoun "{span2.lower()}" refer to {span1}? Yes or No?\n'

templates = {
    'boolq': BoolQTemplate,
    'rte': RTETemplate,
    'multirc': MultiRCTemplate,
    'cb': CBTemplate,
    'wic': WICTemplate,
    'wsc': WSCTemplate
}

def load_superglue_data(dataset_name, tokenizer):
    from datasets import load_dataset

    dataset = load_dataset('super_glue', dataset_name)

    sentence_keys = set(dataset['train'].column_names) - {'label', 'idx'}
    seq_len = 512
    if dataset_name == 'cb':
        seq_len = 128
    elif dataset_name == 'rte':
        seq_len = 128
    elif dataset_name == 'wsc':
        seq_len = 128
    elif dataset_name == 'wic':
        seq_len = 128
    elif dataset_name == 'boolq':
        seq_len = 512

    def tokenize_function(examples):
        batch_size = len(examples['label'])
        texts = [templates[dataset_name]({k: examples[k][i] for k in examples}) for i in range(batch_size)]
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            return_tensors='pt'
        )
    # Tokenize datasets
    train_ds = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=list(sentence_keys)
    )
    test_ds = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=list(sentence_keys)
    )

    # Convert to PyTorch format
    train_ds.set_format('torch')
    test_ds.set_format('torch')

    return train_ds, test_ds

def load_ag_news_data(tokenizer):
    from datasets import load_dataset

    dataset = load_dataset('ag_news')

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

    # Tokenize datasets
    train_ds = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text']  # Only remove the sentence column
    )
    test_ds = dataset['test'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text']  # Only remove the sentence column
    )

    # Convert to PyTorch format
    train_ds.set_format('torch')
    test_ds.set_format('torch')

    return train_ds, test_ds

def record_net_data_stats(y_train, net_dataidx_map, logdir=None):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # if logdir != None:
    #     logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, partition, n_parties, beta=0.4, logdir=None, args=None):
    if dataset == 'sst2':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        train_ds, test_ds = load_sst2_data(tokenizer)
        K = 2  # Binary classification
    elif 'superglue' in dataset:
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        dataset_name = dataset.split('_')[-1]
        train_ds, test_ds = load_superglue_data(dataset_name, tokenizer)
        if dataset_name == 'cb':
            K = 3
        else:
            K = 2
    elif dataset == 'ag_news':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        train_ds, test_ds = load_ag_news_data(tokenizer)
        K = 4
    elif dataset == 'cifar10':
        train_ds, test_ds = load_cifar10_data(datadir)
        K = 10
    elif dataset == 'cifar100':
        train_ds, test_ds = load_cifar100_data(datadir)
        K = 100
    elif dataset == 'tinyimagenet':
        train_ds, test_ds = load_tinyimagenet_data(datadir)
        K = 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if dataset == 'sst2' or dataset == 'ag_news' or 'superglue' in dataset:
        # For SST-2, we'll use the label field directly
        y_train = train_ds['label']
        y_test = test_ds['label']
    else:
        # For image datasets, use the existing logic
        X_train, y_train = train_ds.data, train_ds.targets
        X_test, y_test = test_ds.data, test_ds.targets

    if type(y_train) != np.ndarray:
        y_train = np.array(y_train)
    if type(y_test) != np.ndarray:
        y_test = np.array(y_test)

    if partition == "noniid-labeluni":
        num = args.cls_num
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10

        N = y_train.shape[0]
        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        idx_batch = [[] for _ in range(n_parties)]
        test_idx_batch = [[] for _ in range(n_parties)]

        while min_size < min_require_size:
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]
                np.random.shuffle(idx_k)
                np.random.shuffle(test_idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                test_proportions = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_batch, np.split(test_idx_k, test_proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map_train[j] = idx_batch[j]
            np.random.shuffle(test_idx_batch[j])
            net_dataidx_map_test[j] = test_idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)

    return (train_ds, test_ds, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)
