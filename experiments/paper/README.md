# Paper experiment drivers

All drivers preview their commands unless `--execute` is supplied. Outputs are stored below `artifacts/`, which is ignored by Git.

## Primary three-seed evaluation

```bash
python3 experiments/paper/three_seed_runner.py \
  --datasets cifar100 tinyimagenet --seeds 0 1 2 \
  --local-policy epoch --gpus 0 1 --execute
```

Use `--local-policy k5` for the five-local-step configuration. Summarize only completed runs with:

```bash
python3 experiments/paper/summarize_three_seeds.py \
  --input-root artifacts/three_seeds --local-policy epoch
```

## Exact Replay

```bash
python3 experiments/paper/exact_replay_runner.py \
  --datasets cifar100 tinyimagenet --modes k5 full_epoch --seeds 0 1 2 \
  --comm-rounds 100 --gpus 0 1 --asset-root . \
  --output-root artifacts/exact_replay --execute
python3 experiments/paper/summarize_exact_replay.py \
  --replay-root artifacts/exact_replay \
  --direct-root artifacts/three_seeds
```

The summary reports reconstruction error, payload, accuracy, and runtime relative to the matching Direct run. A result is valid only when parameter and momentum reconstruction errors pass the configured numerical tolerance.

## Client parallelism

```bash
python3 experiments/paper/client_parallel_runner.py \
  --dataset cifar100 --data-dir data --checkpoint checkpoints/imagenet21k_ViT-B_16.npz \
  --output-root artifacts/client_parallel/cifar100/repeat-0/two-worker \
  --gpus 0 1 --num-workers 2 --execute
python3 experiments/paper/summarize_client_parallel.py \
  --root artifacts/client_parallel \
  --direct-reference-status passed
```

The runtime benchmark compares sequential and two-GPU execution of independent selected clients. Treat deterministic-equivalence validation and timing as separate checks; a timing result alone does not establish semantic equivalence.
