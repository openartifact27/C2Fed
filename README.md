# C2Fed

C2Fed is a communication--computation co-design for federated fine-tuning without backbone backpropagation. It updates the classification head analytically and applies uncertainty-guided zeroth-order (ZO) estimation only to compact parameter-efficient fine-tuning (PEFT) parameters.

This repository is the anonymous artifact accompanying the submitted paper. The default path reproduces the paper's **C2Fed-Direct** protocol; **C2Fed-Replay** is an optional bandwidth-oriented variant.

## Repository layout

- `c2fed.py`: federated training entry point.
- `local_training.py`: analytical head update and uncertainty-guided ZO training.
- `federated_protocol.py`: Direct and Replay protocol accounting/reconstruction.
- `experiments/paper/`: paper experiment runners and result summarizers.
- `parallel_client_runtime.py`: one- and two-GPU client execution runtime.
- `tests/`: unit and integration tests.
- `scripts/`: compact vision and NLP launch examples.

## Environment

The artifact requires Python 3.10 or newer, PyTorch with CUDA support, and the packages in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

Datasets are downloaded by the training code when supported by the upstream dataset package. Put the pretrained ViT checkpoint at `checkpoints/imagenet21k_ViT-B_16.npz`; use `python3 c2fed.py --help` to inspect all paths and options.

## Quick start

Run the primary vision configuration (CIFAR-100, BitFit, one local epoch):

```bash
bash scripts/run_vision.sh cifar100 epoch 0
```

Use `tinyimagenet` for the second many-class task or `k5` for the lower-compute five-local-step setting.

## Paper experiments

The runners default to preview mode. Add `--execute` only after checking the printed commands and paths.

Three-seed Direct evaluation:

```bash
python3 experiments/paper/three_seed_runner.py \
  --datasets cifar100 tinyimagenet --seeds 0 1 2 --gpus 0 1

python3 experiments/paper/three_seed_runner.py \
  --datasets cifar100 tinyimagenet --seeds 0 1 2 --gpus 0 1 --execute

python3 experiments/paper/summarize_three_seeds.py \
  --input-root artifacts/three_seeds --local-policy epoch
```

Exact-Replay evaluation:

```bash
python3 experiments/paper/exact_replay_runner.py \
  --datasets cifar100 tinyimagenet --modes k5 full_epoch --seeds 0 1 2 \
  --comm-rounds 100 --gpus 0 1 --asset-root . \
  --output-root artifacts/exact_replay
python3 experiments/paper/exact_replay_runner.py \
  --datasets cifar100 tinyimagenet --modes k5 full_epoch --seeds 0 1 2 \
  --comm-rounds 100 --gpus 0 1 --asset-root . \
  --output-root artifacts/exact_replay --execute
python3 experiments/paper/summarize_exact_replay.py
```

Client-parallel execution:

```bash
python3 experiments/paper/client_parallel_runner.py \
  --dataset cifar100 --data-dir data --checkpoint checkpoints/imagenet21k_ViT-B_16.npz \
  --output-root artifacts/client_parallel/cifar100/repeat-0/two-worker \
  --gpus 0 1 --num-workers 2
python3 experiments/paper/client_parallel_runner.py \
  --dataset cifar100 --data-dir data --checkpoint checkpoints/imagenet21k_ViT-B_16.npz \
  --output-root artifacts/client_parallel/cifar100/repeat-0/two-worker \
  --gpus 0 1 --num-workers 2 --execute
python3 experiments/paper/summarize_client_parallel.py \
  --root artifacts/client_parallel --direct-reference-status passed
```

See `experiments/paper/README.md` for the output layout and validity checks.

## Reproducibility notes

- **Direct protocol:** exchanges the compact PEFT state and client momentum in both directions, for a logical payload of `4S` per selected client and round. The frozen backbone is not included.
- **Replay protocol:** communicates compact state and head state plus a seed/derivative trace. Payload accounting excludes one-time model provisioning and transport framing.
- **Aggregation:** the evaluated experiments use equal weighting across selected clients.
- **ZO convention:** the released experiments perturb parameters by `h v`, compute `(L(theta+h v)-L(theta))/h`, and accumulate the fixed-scale update `(L(theta+h v)-L(theta))v` before clipping. This convention is preserved to reproduce the reported trajectories.
- Random seeds, completed-run markers, resolved arguments, logs, and summaries are written under `artifacts/`.

## Validation

Run the CPU-safe test suite before launching experiments:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m unittest discover -s tests -v
```

## Anonymous release

Author and citation metadata are intentionally omitted during double-blind review. They will be added after the review process.
