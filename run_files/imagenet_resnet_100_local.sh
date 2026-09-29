#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python main_local_energy.py --data "${DATA_ROOT:-./dataset/ImageNet2012}"  \
  --lr 0.1 -p 300 --epochs 100 \
  --arch resnet50 \
  --use_norm \
  --wd 5e-4 \
  --cos \
  --cl_views sim-sim\
  --batch-size 256\
  --tau 0.99\
  --root_log "${LOG_ROOT:-./logs}"\
  --l_d_warm 70\
  --scaling_factor 200 255 \
  --topk 30 "$@"
