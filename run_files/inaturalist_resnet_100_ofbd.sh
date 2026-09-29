#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python main_ofbd_rl.py --data "${DATA_ROOT:-./dataset/train_val2018}"   --dataset inat\
  --lr 0.2 -p 600 --epochs 100 \
  --arch resnet50 \
  --use_norm  \
  --wd 1e-4 \
  --cos \
  --cl_views sim-sim\
  --batch-size 128\
  --tau 0.99\
  --root_log "${LOG_ROOT:-./logs}"\
  --l_d_warm 80\
  --scaling_factor 1628 255 \
  --topk 30\
  --grad_c\
  --paper_rl_m 4 \
  --paper_rl_k 2 \
  --paper_rl_prob 0.1 \
  --paper_rl_warmup_epochs 30 \
  --paper_rl_energy_weight 10 \
  --paper_rl_residual_weight 10 \
  --paper_rl_lr 3e-4 \
  --paper_rl_kl 0.02 \
  --paper_selector_arch lite "$@"
