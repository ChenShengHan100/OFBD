# OFBD

OFBD is a long-tailed image recognition codebase with foreground/background-aware feature aggregation and CutMix training. The active implementation focuses on CIFAR-LT with ResNet-32 and supports local-energy and reinforcement-learning region selection.

[Paper](OFBD_Paper.pdf) · [Code provenance](ACKNOWLEDGEMENTS.md)

The training framework, dataset loaders, backbones, and losses are included directly in this repository. No separate ConCutMix checkout is required.

## What Is Implemented

- **OFBD / BGE inference head**: `models/resnet32.py` uses `OFBDAggregationHead`, which estimates foreground/background masks from the final feature map and classifies with `fg_feat - bg_scale * bg_feat`.
- **Local-energy CutMix**: `main_local_energy.py` selects the donor patch around the foreground-energy peak from `model.last_soft_weights`.
- **RL residual selector**: `main_ofbd_rl.py` uses `EnergyPriorMultiBoxHelper` to sample `M` proposals, select `K` regions, and train a lightweight selector as a residual correction over the local-energy prior.

The active training path is **CIFAR-10/100-LT + ResNet-32**. ImageNet-LT and iNaturalist launch scripts and a Places-LT loader are retained, but need validation before reporting new results.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `main_local_energy.py` | OFBD with local-energy guided CutMix |
| `main_ofbd_rl.py` | OFBD with paper-style RL residual region selector |
| `models/resnet32.py` | ResNet-32 backbone, BGE/OFBD aggregation head, auxiliary feature-layer access |
| `models/paper_prior_rl.py` | Feature proposal selector and energy-prior RL helper |
| `models/multibox_k_cutmix.py` | Multi-box local-energy and RL CutMix helpers |
| `dataset/cifar.py` | CIFAR-LT dataset construction |
| `loss/contrastive.py` | Balanced supervised contrastive loss |
| `loss/logitadjust.py` | Logit-adjusted CE and soft-label CutMix CE |
| `run_files/` | Bash launch scripts for common datasets/settings |
| `plot/` | Hessian and loss-landscape analysis tools |
| `tools/` | Selector-layer evaluation utilities |
| `third_party/` | Optional comparison repositories, pinned as submodules |

The model APIs are `OFBDModel32` and `OFBDModel`. The training objectives are `OFBDContrastiveLoss`, `OFBDLogitAdjustLoss`, and `OFBDCutMixLoss`. Model parameter names are preserved so that renaming the Python classes does not change the saved `state_dict` format.

## Environment

Clone the repository and install the dependencies in a Python 3.9 environment:

```bash
git clone https://github.com/ChenShengHan100/OFBD.git
cd OFBD
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency versions in `requirements.txt` match the local training environment, which uses PyTorch 2.8.0 and torchvision 0.23.0. Training requires a CUDA-enabled PyTorch installation compatible with your GPU. The existing environment used a CUDA 12.8 build.

Neptune logging is disabled by default and is imported only when `--logger neptune` is selected.

The optional comparison repositories can be fetched separately:

```bash
git submodule update --init --recursive
```

They are not needed by either OFBD training entry point and have their own dependencies.

## Dataset

CIFAR-10 and CIFAR-100 are loaded through torchvision and can be downloaded automatically into `./dataset`. Dataset images, archives, model weights, and generated experiment outputs are not included. Dataset split lists are included alongside their loaders. Supply your own data root for ImageNet-LT, Places-LT, and iNaturalist.

For CIFAR-LT, imbalance factor is passed directly:

| Imbalance ratio | Argument |
| ---: | --- |
| 10 | `--imb_factor 0.1` |
| 50 | `--imb_factor 0.02` |
| 100 | `--imb_factor 0.01` |

## Training

Run the following commands from the repository root.

### CIFAR100-LT IF100, Local-Energy OFBD

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_local_energy.py \
  --data ./dataset \
  --dataset cifar100 --num_classes 100 --imb_factor 0.01 \
  --arch resnet32 --epochs 200 --batch-size 256 --lr 0.15 -p 194 --wd 5e-4 \
  --cl_views uncutout-sim --warmup_epochs 5 --feat_dim 128 \
  --alpha 2 --beta 0.6 --temp 0.1 --tau 0.9 \
  --cutmix_prob 0.5 --l_d_warm 100 --topk 30 \
  --scaling_factor 20 255 --device_ids 0 --seed 3407 \
  --root_log logs --file_name LOCAL_if100
```

Preset script (pass additional arguments to override defaults):

```bash
bash run_files/cifar100_imb100_local.sh --seed 3407 --device_ids 0
```

Launch scripts locate the repository root automatically. Set `DATA_ROOT` and `LOG_ROOT` to customize paths, for example `DATA_ROOT=/path/to/data LOG_ROOT=./logs bash run_files/cifar100_imb100_local.sh`.

The number of classes is inferred from `--dataset` unless `--num_classes` is supplied, and training creates the output directory automatically.

### CIFAR100-LT IF100, OFBD + RL Selector

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_ofbd_rl.py \
  --data ./dataset \
  --dataset cifar100 --num_classes 100 --imb_factor 0.01 \
  --arch resnet32 --epochs 200 --batch-size 256 --lr 0.15 -p 194 --wd 5e-4 \
  --cl_views uncutout-sim --warmup_epochs 5 --feat_dim 128 \
  --alpha 2 --beta 0.6 --temp 0.1 --tau 0.9 \
  --cutmix_prob 0.5 --l_d_warm 100 --topk 30 \
  --scaling_factor 20 255 --device_ids 0 --seed 3407 \
  --paper_rl_m 4 --paper_rl_k 2 --paper_rl_prob 0.01 \
  --paper_rl_warmup_epochs 30 \
  --paper_rl_energy_weight 20 --paper_rl_residual_weight 0.1 \
  --paper_rl_lr 5e-5 --paper_rl_kl 0.02 \
  --paper_selector_arch lite \
  --root_log logs --file_name OFBD_RL_if100
```

Preset script:

```bash
bash run_files/cifar100_imb100_ofbd.sh --seed 3407 --device_ids 0
```

## Evaluation And Resume

Evaluate a saved checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_local_energy.py \
  --data ./dataset \
  --dataset cifar100 --num_classes 100 --imb_factor 0.01 \
  --arch resnet32 --batch-size 256 \
  --resume /path/to/OFBD_ckpt.best.pth.tar \
  --reload True
```

Resume the automatically named run directory:

```bash
python -u main_local_energy.py ... --auto_resume
```

Checkpoints are written as:

```text
<root_log>/<store_name>/OFBD_ckpt.pth.tar
<root_log>/<store_name>/OFBD_ckpt.best.pth.tar
```

## Important Notes

- Inference uses the OFBD/BGE aggregation head.  RL selector and local-energy CutMix are training-time augmentation mechanisms.
- CIFAR-LT is the most reliable tested path in this workspace.  Revalidate large-scale dataset scripts before using them for final tables.
- `models/ofbd_mixup.py` is currently a placeholder class; the active CutMix behavior is implemented directly in the training loops and helper modules.

## Attribution And Licensing

OFBD incorporates code from [ConCutMix](https://github.com/PanHaulin/ConCutMix). See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for the source revision, citation, and licensing status. The imported revision has no license file, so this repository does not apply a new project-wide license to the upstream code. Optional third-party repositories retain their own license terms.
