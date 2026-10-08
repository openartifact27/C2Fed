#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-cifar100}"
local_policy="${2:-epoch}"
seed="${3:-0}"

case "${dataset}" in
  cifar100|tinyimagenet) ;;
  *) echo "dataset must be cifar100 or tinyimagenet" >&2; exit 2 ;;
esac

case "${local_policy}" in
  epoch) local_steps=0 ;;
  k5) local_steps=5 ;;
  *) echo "local policy must be epoch or k5" >&2; exit 2 ;;
esac

python3 c2fed.py \
  --model_name ViT-B_16 \
  --dataset "${dataset}" --partition noniid-labeluni --cls_num 50 \
  --n_parties 100 --sample 0.05 --comm_round 100 --test_round 99 \
  --lr 0.1 --head_lr 0.1 -K "${local_steps}" --batch_size 64 --device cuda \
  --train fwd --fwd_head_tuning --h 0.001 --clip_norm 1 \
  --config_profile c2fed-direct --momentum_type direct --momentum 0.9 \
  -N 8 --micro_batch_size 8 --num_micro_batch 2 \
  --sample_selection uncertainty --peftmode bitfit --init_seed "${seed}" \
  --logdir "./artifacts/manual/${local_policy}/${dataset}/seed-${seed}/"
