#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python main_ofbd_rl.py --data "${DATA_ROOT:-./dataset/ImageNet2012}"  \
  --lr 0.1 -p 300 --epochs 90 \
  --arch resnext50 \
  --use_norm \
  --wd 5e-4 \
  --cos \
  --cl_views sim-sim\
  --batch-size 256\
  --tau 0.99\
  --root_log "${LOG_ROOT:-./logs}"\
  --l_d_warm 60\
  --scaling_factor 200 255 \
  --topk 30\
  --paper_rl_m 8 \
  --paper_rl_k 2 \
  --paper_rl_prob 0.1 \
  --paper_rl_warmup_epochs 30 \
  --paper_rl_energy_weight 10 \
  --paper_rl_residual_weight 10 \
  --paper_rl_lr 1e-4 \
  --paper_rl_kl 0.02 \
  --paper_selector_arch lite "$@"
