import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from plot.analyze_hessian_ce import (
    build_cls_num_list,
    build_eval_loader_for_classes,
    build_model,
    build_split_class_groups,
    get_device,
    iter_batches,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D CE loss landscape around a standalone ResNet32 checkpoint."
    )
    parser.add_argument(
        "--ckpt",
        default="./log_sup/standalone_pure_ce_resnet32_cifar100_resnet32_batchsize_256_epochs_200_lr_0.1_wd_0.0005_if_0.01_seed_2058833602/ckpt_best.pth.tar",
    )
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--subset-size", default=1024, type=int)
    parser.add_argument("--max-batches", default=4, type=int)
    parser.add_argument("--split", default="all", choices=["all", "head", "medium", "tail"])
    parser.add_argument("--grid-points", default=21, type=int)
    parser.add_argument("--x-min", default=-0.1, type=float)
    parser.add_argument("--x-max", default=0.1, type=float)
    parser.add_argument("--y-min", default=-0.1, type=float)
    parser.add_argument("--y-max", default=0.1, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu", default=7, type=int)
    parser.add_argument(
        "--output-dir",
        default="./log_sup/standalone_ce_loss_landscape",
    )
    return parser.parse_args()


def collect_named_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def clone_state(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def sample_direction_like(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    seed: int,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device=named_params[0][1].device)
    gen.manual_seed(seed)
    direction = {}
    for name, param in named_params:
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param_norm = param.detach().norm()
        noise_norm = noise.norm().clamp_min(1e-12)
        if param_norm.item() == 0.0:
            direction[name] = torch.zeros_like(param)
        else:
            direction[name] = noise * (param_norm / noise_norm)
    return direction


def orthogonalize_direction(
    base_dir: Dict[str, torch.Tensor],
    other_dir: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    dot = sum(torch.sum(base_dir[k] * other_dir[k]) for k in base_dir.keys())
    base_norm_sq = sum(torch.sum(base_dir[k] * base_dir[k]) for k in base_dir.keys()).clamp_min(1e-12)
    coeff = dot / base_norm_sq
    return {k: other_dir[k] - coeff * base_dir[k] for k in other_dir.keys()}


def apply_perturbation(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    base_state: Dict[str, torch.Tensor],
    dir_x: Dict[str, torch.Tensor],
    dir_y: Dict[str, torch.Tensor],
    x: float,
    y: float,
) -> None:
    with torch.no_grad():
        for name, param in named_params:
            param.copy_(base_state[name] + x * dir_x[name] + y * dir_y[name])


def evaluate_point(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in batches:
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += int(targets.size(0))
    return {
        "loss": total_loss / total_examples,
        "top1": 100.0 * total_correct / total_examples,
    }


def plot_surfaces(
    xs: np.ndarray,
    ys: np.ndarray,
    losses: np.ndarray,
    accs: np.ndarray,
    output_prefix: str,
) -> None:
    xgrid, ygrid = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(xgrid, ygrid, losses, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title("Loss Landscape")
    ax1.set_xlabel("Direction X")
    ax1.set_ylabel("Direction Y")
    ax1.set_zlabel("Loss")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1)

    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(xgrid, ygrid, losses, levels=30, cmap="viridis")
    ax2.set_title("Loss Contour")
    ax2.set_xlabel("Direction X")
    ax2.set_ylabel("Direction Y")
    fig.colorbar(contour, ax=ax2)
    fig.tight_layout()
    fig.savefig(output_prefix + "_loss.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(xgrid, ygrid, accs, cmap="plasma", linewidth=0, antialiased=True)
    ax1.set_title("Accuracy Landscape")
    ax1.set_xlabel("Direction X")
    ax1.set_ylabel("Direction Y")
    ax1.set_zlabel("Top1")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1)

    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(xgrid, ygrid, accs, levels=30, cmap="plasma")
    ax2.set_title("Accuracy Contour")
    ax2.set_xlabel("Direction X")
    ax2.set_ylabel("Direction Y")
    fig.colorbar(contour, ax=ax2)
    fig.tight_layout()
    fig.savefig(output_prefix + "_acc.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu)
    checkpoint = load_checkpoint(args.ckpt, device)
    ckpt_args = checkpoint.get("args", {})
    num_classes = int(ckpt_args.get("num_classes", 100))
    cls_num_list = build_cls_num_list(checkpoint, num_classes, args.data)
    class_groups = build_split_class_groups(cls_num_list)
    classes = class_groups[args.split]

    loader = build_eval_loader_for_classes(
        data_root=args.data,
        classes=classes,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=args.subset_size,
    )
    batches = list(iter_batches(loader, device, args.max_batches))
    if not batches:
        raise RuntimeError("No batches available for loss landscape evaluation.")

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    criterion = nn.CrossEntropyLoss().to(device)

    named_params = collect_named_params(model)
    base_state = clone_state(named_params)
    dir_x = sample_direction_like(named_params, args.seed)
    dir_y_raw = sample_direction_like(named_params, args.seed + 1)
    dir_y = orthogonalize_direction(dir_x, dir_y_raw)

    xs = np.linspace(args.x_min, args.x_max, args.grid_points)
    ys = np.linspace(args.y_min, args.y_max, args.grid_points)
    losses = np.zeros((len(ys), len(xs)), dtype=np.float32)
    accs = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            apply_perturbation(named_params, base_state, dir_x, dir_y, float(x), float(y))
            stats = evaluate_point(model, criterion, batches)
            losses[iy, ix] = stats["loss"]
            accs[iy, ix] = stats["top1"]

    apply_perturbation(named_params, base_state, dir_x, dir_y, 0.0, 0.0)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_tag = os.path.basename(os.path.dirname(args.ckpt)) or "checkpoint"
    prefix = os.path.join(args.output_dir, f"{ckpt_tag}_eval_ce_{args.split}")

    plot_surfaces(xs, ys, losses, accs, prefix)
    np.savez_compressed(
        prefix + "_grid.npz",
        xs=xs,
        ys=ys,
        losses=losses,
        accs=accs,
    )

    center_idx = args.grid_points // 2
    summary = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "subset_size": args.subset_size,
        "max_batches": args.max_batches,
        "grid_points": args.grid_points,
        "x_range": [args.x_min, args.x_max],
        "y_range": [args.y_min, args.y_max],
        "min_loss": float(losses.min()),
        "max_loss": float(losses.max()),
        "center_loss": float(losses[center_idx, center_idx]),
        "center_top1": float(accs[center_idx, center_idx]),
    }
    with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
