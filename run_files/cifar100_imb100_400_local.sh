#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python main_local_energy.py\
  --data "${DATA_ROOT:-./dataset}" \
  --lr 0.15 -p 194 --epochs 400 \
  --arch resnet32 \
  --wd 5e-4 \
  --cl_views uncutout-sim \
  --batch-size 256\
  --warmup_epochs 10\
  --feat_dim 128\
  --alpha 2\
  --beta 0.6\
  --temp 0.1\
  --tau 0.79\
  --root_log "${LOG_ROOT:-./logs}"\
  --dataset cifar100\
  --imb_factor 0.01\
  --l_d_warm 160\
  --topk 30\
  --scaling_factor 20 255\
  --schedule 360 380 "$@"
