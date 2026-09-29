#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python main_ofbd_rl.py\
  --data "${DATA_ROOT:-./dataset}" \
  --lr 0.15 -p 34 --epochs 200 \
  --arch resnet32 \
  --wd 5e-4 \
  --cl_views uncutout-sim \
  --batch-size 256\
  --warmup_epochs 5\
  --feat_dim 128\
  --alpha 2 \
  --beta 0.6\
  --temp 0.1\
  --tau 0.85\
  --root_log "${LOG_ROOT:-./logs}"\
  --dataset cifar10\
  --imb_factor 0.01\
  --l_d_warm 100\
  --topk 3\
  --scaling_factor 2 255\
  --paper_rl_m 4 \
  --paper_rl_k 2 \
  --paper_rl_prob 0.1 \
  --paper_rl_warmup_epochs 30 \
  --paper_rl_energy_weight 10 \
  --paper_rl_residual_weight 10 \
  --paper_rl_lr 3e-4 \
  --paper_rl_kl 0.02 \
  --paper_selector_arch lite "$@"
