import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_hessian_main1 import (
    build_dataloaders,
    build_eval_loader_for_classes,
    build_criterion,
    build_model,
    build_scl_criterion,
    build_split_class_groups,
    build_train_loader_for_total_loss,
    evaluate_loss_and_acc,
    evaluate_total_train_loss_and_acc,
    get_device,
    iter_batches,
    iter_train_total_batches,
    load_checkpoint_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D loss landscape around a OFBD checkpoint."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--feat-dim", default=128, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--imb-factor", default=0.01, type=float)
    parser.add_argument("--tau", default=0.9, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=0.0, type=float)
    parser.add_argument(
        "--ce-loss-type",
        default="cross_entropy",
        choices=["cross_entropy", "logit_adjust"],
    )
    parser.add_argument(
        "--loss-mode",
        default="eval_ce",
        choices=["eval_ce", "train_total"],
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "head", "medium", "tail"],
    )
    parser.add_argument("--subset-size", default=1024, type=int)
    parser.add_argument("--max-batches", default=4, type=int)
    parser.add_argument("--grid-points", default=21, type=int)
    parser.add_argument("--x-min", default=-0.5, type=float)
    parser.add_argument("--x-max", default=0.5, type=float)
    parser.add_argument("--y-min", default=-0.5, type=float)
    parser.add_argument("--y-max", default=0.5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default="./log_sup/loss_landscape",
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
    ortho = {k: other_dir[k] - coeff * base_dir[k] for k in other_dir.keys()}
    return ortho


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


def prepare_batches(args: argparse.Namespace, device: torch.device):
    train_dataset, val_dataset = build_dataloaders(
        data_root=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=args.subset_size,
        imb_factor=args.imb_factor,
    )
    class_groups = build_split_class_groups(train_dataset.cls_num_list)
    split_classes = class_groups[args.split]

    if args.loss_mode == "eval_ce":
        loader = build_eval_loader_for_classes(
            val_dataset=val_dataset,
            classes=split_classes,
            batch_size=args.batch_size,
            workers=args.workers,
            subset_size=args.subset_size,
        )
        batches = list(iter_batches(loader, device, args.max_batches))
        return train_dataset, batches

    train_dataset_total, train_loader = build_train_loader_for_total_loss(
        data_root=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=args.subset_size,
        imb_factor=args.imb_factor,
    )
    selected_indices = [
        idx for idx, target in enumerate(train_dataset_total.targets) if int(target) in set(split_classes)
    ][: args.subset_size]
    subset = torch.utils.data.Subset(train_dataset_total, selected_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    batches = list(iter_train_total_batches(loader, device, args.max_batches))
    return train_dataset_total, batches


def evaluate_point(
    model: torch.nn.Module,
    train_dataset,
    batches,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    if args.loss_mode == "eval_ce":
        criterion = build_criterion(
            cls_num_list=train_dataset.cls_num_list,
            tau=args.tau,
            ce_loss_type=args.ce_loss_type,
            device=device,
        )
        loss_value, top1 = evaluate_loss_and_acc(model, criterion, batches)
        return {"loss": loss_value, "top1": top1}

    criterion_ce = build_criterion(
        cls_num_list=train_dataset.cls_num_list,
        tau=args.tau,
        ce_loss_type=args.ce_loss_type,
        device=device,
    )
    criterion_scl = build_scl_criterion(train_dataset.cls_num_list, temp=0.1, device=device)
    loss_value, top1 = evaluate_total_train_loss_and_acc(
        model, criterion_ce, criterion_scl, batches, args.alpha, args.beta
    )
    return {"loss": loss_value, "top1": top1}


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
    model = build_model(args.num_classes, args.feat_dim, device)
    state_dict = load_checkpoint_state(args.ckpt, device)
    model.load_state_dict(state_dict, strict=False)

    train_dataset, batches = prepare_batches(args, device)
    if not batches:
        raise RuntimeError("No batches available for loss landscape evaluation.")

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
            stats = evaluate_point(model, train_dataset, batches, args, device)
            losses[iy, ix] = stats["loss"]
            accs[iy, ix] = stats["top1"]

    apply_perturbation(named_params, base_state, dir_x, dir_y, 0.0, 0.0)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_tag = os.path.basename(os.path.dirname(args.ckpt)) or "checkpoint"
    prefix = os.path.join(args.output_dir, f"{ckpt_tag}_{args.loss_mode}_{args.split}")

    np.savez_compressed(
        prefix + "_grid.npz",
        xs=xs,
        ys=ys,
        losses=losses,
        accs=accs,
    )
    plot_surfaces(xs, ys, losses, accs, prefix)

    summary = {
        "ckpt": args.ckpt,
        "loss_mode": args.loss_mode,
        "split": args.split,
        "grid_points": args.grid_points,
        "x_range": [args.x_min, args.x_max],
        "y_range": [args.y_min, args.y_max],
        "min_loss": float(losses.min()),
        "max_loss": float(losses.max()),
        "center_loss": float(losses[len(ys) // 2, len(xs) // 2]),
        "center_top1": float(accs[len(ys) // 2, len(xs) // 2]),
        "output_prefix": prefix,
    }
    with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved plots to: {prefix}_loss.png and {prefix}_acc.png")


if __name__ == "__main__":
    main()
