import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

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
    collect_trainable_params,
    dot_tensors,
    get_device,
    hessian_vector_product,
    iter_batches,
    l2_norm_tensors,
    load_checkpoint,
    make_rademacher_like,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate full Hessian spectral density with stochastic Lanczos quadrature."
    )
    parser.add_argument(
        "--ckpt",
        default="./log_sup/standalone_pure_ce_resnet32_cifar100_resnet32_batchsize_256_epochs_200_lr_0.1_wd_0.0005_if_0.01_seed_2058833602/ckpt_best.pth.tar",
    )
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--subset-size", default=1024, type=int)
    parser.add_argument("--max-batches", default=4, type=int)
    parser.add_argument("--split", default="all", choices=["all", "head", "medium", "tail"])
    parser.add_argument("--num-probes", default=10, type=int)
    parser.add_argument("--lanczos-steps", default=40, type=int)
    parser.add_argument("--density-points", default=800, type=int)
    parser.add_argument("--plot-points", default=80, type=int)
    parser.add_argument("--sigma-ratio", default=0.03, type=float)
    parser.add_argument("--x-min", default=None, type=float)
    parser.add_argument("--x-max", default=None, type=float)
    parser.add_argument("--focus-threshold", default=None, type=float)
    parser.add_argument("--focus-points", default=160, type=int)
    parser.add_argument("--tail-points", default=40, type=int)
    parser.add_argument("--segment-neg-points", default=30, type=int)
    parser.add_argument("--segment-mid-points", default=260, type=int)
    parser.add_argument("--segment-pos1-points", default=60, type=int)
    parser.add_argument("--segment-pos2-points", default=40, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument(
        "--output-prefix",
        default="./log_sup/standalone_ce_hessian_slq_all",
    )
    return parser.parse_args()


def clone_tensors(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.detach().clone() for x in xs]


def normalize_vector(xs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
    norm = float(l2_norm_tensors(xs).item())
    denom = max(norm, 1e-12)
    return [x / denom for x in xs], norm


def lanczos_tridiagonal(
    model: nn.Module,
    criterion: nn.Module,
    batches,
    params: List[torch.nn.Parameter],
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    q, _ = normalize_vector(make_rademacher_like(params))
    q_prev = [torch.zeros_like(param) for param in params]
    beta_prev = 0.0
    q_basis: List[List[torch.Tensor]] = []
    alphas: List[float] = []
    betas: List[float] = []

    for step in range(steps):
        q_basis.append(clone_tensors(q))
        z = hessian_vector_product(model, criterion, batches, params, q)
        if step > 0:
            z = [zi - beta_prev * qi for zi, qi in zip(z, q_prev)]

        alpha = float(dot_tensors(q, z).item())
        z = [zi - alpha * qi for zi, qi in zip(z, q)]

        # Full re-orthogonalization for numerical stability.
        for basis_vec in q_basis:
            coeff = float(dot_tensors(z, basis_vec).item())
            z = [zi - coeff * bi for zi, bi in zip(z, basis_vec)]

        beta = float(l2_norm_tensors(z).item())
        alphas.append(alpha)
        if step < steps - 1:
            betas.append(beta)
        if beta < 1e-10:
            break

        q_prev = clone_tensors(q)
        q = [zi / beta for zi in z]
        beta_prev = beta

    return np.asarray(alphas, dtype=np.float64), np.asarray(betas, dtype=np.float64)


def tridiagonal_matrix(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    n = len(alphas)
    tri = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        tri[i, i] = alphas[i]
        if i < n - 1 and i < len(betas):
            tri[i, i + 1] = betas[i]
            tri[i + 1, i] = betas[i]
    return tri


def slq_density(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    xs: np.ndarray,
    sigma: float,
) -> np.ndarray:
    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    ys = np.zeros_like(xs)
    for eig, weight in zip(eigenvalues, weights):
        z = (xs - eig) / sigma
        ys += weight * np.exp(-0.5 * z * z) * coeff
    return ys


def build_plot_grid(
    x_min: float,
    x_max: float,
    num_points: int,
    focus_threshold: Optional[float],
    focus_points: int,
    tail_points: int,
) -> np.ndarray:
    if focus_threshold is None or focus_threshold <= x_min or focus_threshold >= x_max:
        return np.linspace(x_min, x_max, num_points)

    left = np.linspace(x_min, focus_threshold, max(focus_points, 2), endpoint=False)
    right = np.linspace(focus_threshold, x_max, max(tail_points, 2))
    xs = np.concatenate([left, right])
    return np.unique(xs)


def build_segmented_grid(
    neg_min: float,
    neg_max: float,
    mid_min: float,
    mid_max: float,
    pos1_max: float,
    pos2_max: float,
    neg_points: int,
    mid_points: int,
    pos1_points: int,
    pos2_points: int,
) -> np.ndarray:
    parts = []
    parts.append(np.linspace(neg_min, neg_max, max(neg_points, 2), endpoint=False))
    parts.append(np.linspace(mid_min, mid_max, max(mid_points, 2), endpoint=False))
    parts.append(np.linspace(mid_max, pos1_max, max(pos1_points, 2), endpoint=False))
    parts.append(np.linspace(pos1_max, pos2_max, max(pos2_points, 2)))
    return np.unique(np.concatenate(parts))


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
        raise RuntimeError("No batches available for SLQ density estimation.")

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    params = collect_trainable_params(model)

    probe_eigs: List[np.ndarray] = []
    probe_weights: List[np.ndarray] = []
    global_min = None
    global_max = None

    for _ in range(args.num_probes):
        alphas, betas = lanczos_tridiagonal(
            model=model,
            criterion=criterion,
            batches=batches,
            params=params,
            steps=args.lanczos_steps,
        )
        tri = tridiagonal_matrix(alphas, betas)
        eigvals, eigvecs = np.linalg.eigh(tri)
        weights = np.square(eigvecs[0, :])
        probe_eigs.append(eigvals)
        probe_weights.append(weights)
        eig_min = float(eigvals.min())
        eig_max = float(eigvals.max())
        global_min = eig_min if global_min is None else min(global_min, eig_min)
        global_max = eig_max if global_max is None else max(global_max, eig_max)

    assert global_min is not None and global_max is not None
    span = max(global_max - global_min, 1e-6)
    sigma = max(span * args.sigma_ratio, 1e-3)
    default_x_min = global_min - 0.05 * span
    default_x_max = global_max + 0.05 * span
    x_min = default_x_min if args.x_min is None else args.x_min
    x_max = default_x_max if args.x_max is None else args.x_max
    if (
        args.x_min is not None
        and args.x_max is not None
        and args.x_min <= -10
        and args.x_max >= 500
    ):
        xs = build_segmented_grid(
            neg_min=-10.0,
            neg_max=-1.0,
            mid_min=-1.0,
            mid_max=10.0,
            pos1_max=100.0,
            pos2_max=500.0,
            neg_points=args.segment_neg_points,
            mid_points=args.segment_mid_points,
            pos1_points=args.segment_pos1_points,
            pos2_points=args.segment_pos2_points,
        )
    else:
        xs = build_plot_grid(
            x_min=x_min,
            x_max=x_max,
            num_points=args.density_points,
            focus_threshold=args.focus_threshold,
            focus_points=args.focus_points,
            tail_points=args.tail_points,
        )
    ys = np.zeros_like(xs)
    for eigvals, weights in zip(probe_eigs, probe_weights):
        ys += slq_density(eigvals, weights, xs, sigma)
    ys /= len(probe_eigs)

    if args.focus_threshold is None:
        plot_idx = np.linspace(0, len(xs) - 1, min(args.plot_points, len(xs)), dtype=int)
        plot_xs = xs[plot_idx]
        plot_ys = ys[plot_idx]
    else:
        plot_xs = xs
        plot_ys = ys

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        plot_xs,
        plot_ys,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=2.8,
    )
    ax.set_xlabel("eig")
    ax.set_ylabel("density")
    ax.set_title(f"SLQ Hessian Eig-Density ({args.split})")
    ax.set_xlim(x_min, x_max)
    if x_min <= -10 and x_max >= 500:
        ax.set_xticks([-10, -1, 1, 10, 100, 500])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    fig.savefig(args.output_prefix + "_density.png", dpi=220)
    plt.close(fig)

    payload = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "subset_size": args.subset_size,
        "max_batches": args.max_batches,
        "num_probes": args.num_probes,
        "lanczos_steps": args.lanczos_steps,
        "plot_points": len(plot_xs),
        "eig_min": global_min,
        "eig_max": global_max,
        "sigma": sigma,
        "focus_threshold": args.focus_threshold,
        "focus_points": args.focus_points,
        "tail_points": args.tail_points,
        "segment_neg_points": args.segment_neg_points,
        "segment_mid_points": args.segment_mid_points,
        "segment_pos1_points": args.segment_pos1_points,
        "segment_pos2_points": args.segment_pos2_points,
        "x_min": float(x_min),
        "x_max": float(x_max),
    }
    with open(args.output_prefix + "_density.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
