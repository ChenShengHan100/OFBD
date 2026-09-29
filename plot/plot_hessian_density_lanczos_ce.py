import argparse
import json
import os
import sys
from typing import List, Tuple

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
    normalize_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Hessian eig-density with Lanczos approximation for standalone CE."
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
    parser.add_argument("--lanczos-steps", default=30, type=int)
    parser.add_argument("--density-points", default=600, type=int)
    parser.add_argument("--bandwidth-ratio", default=0.04, type=float)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument(
        "--output-prefix",
        default="./log_sup/standalone_ce_hessian_lanczos_all",
    )
    return parser.parse_args()


def clone_tensors(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.detach().clone() for x in xs]


def sub_scaled(
    xs: List[torch.Tensor],
    ys: List[torch.Tensor],
    scale: torch.Tensor,
) -> List[torch.Tensor]:
    return [x - scale * y for x, y in zip(xs, ys)]


def add_scaled(
    xs: List[torch.Tensor],
    ys: List[torch.Tensor],
    scale: torch.Tensor,
) -> List[torch.Tensor]:
    return [x + scale * y for x, y in zip(xs, ys)]


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
    q_prev = [torch.zeros_like(p) for p in params]
    beta_prev = torch.tensor(0.0, device=params[0].device)
    alphas: List[float] = []
    betas: List[float] = []

    for step in range(steps):
        z = hessian_vector_product(model, criterion, batches, params, q)
        if step > 0:
            z = sub_scaled(z, q_prev, beta_prev)
        alpha = dot_tensors(q, z)
        z = sub_scaled(z, q, alpha)

        # full reorthogonalization against q and q_prev only is usually enough for short runs
        beta = l2_norm_tensors(z)
        alphas.append(float(alpha.item()))
        beta_value = float(beta.item())
        if step < steps - 1:
            betas.append(beta_value)
        if beta_value < 1e-10:
            break

        q_prev = clone_tensors(q)
        q = [vec / beta for vec in z]
        beta_prev = beta

    return np.asarray(alphas, dtype=np.float64), np.asarray(betas, dtype=np.float64)


def tridiagonal_eigvals(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    n = len(alphas)
    t = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        t[i, i] = alphas[i]
        if i < n - 1 and i < len(betas):
            t[i, i + 1] = betas[i]
            t[i + 1, i] = betas[i]
    return np.linalg.eigvalsh(t)


def kde_on_actual_axis(eigvals: np.ndarray, xs: np.ndarray, bandwidth: float) -> np.ndarray:
    coeff = 1.0 / (len(eigvals) * bandwidth * np.sqrt(2.0 * np.pi))
    ys = np.zeros_like(xs)
    for val in eigvals:
        z = (xs - val) / bandwidth
        ys += np.exp(-0.5 * z * z)
    return coeff * ys


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
        raise RuntimeError("No batches available for Lanczos density estimation.")

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    criterion = nn.CrossEntropyLoss().to(device)
    params = collect_trainable_params(model)

    alphas, betas = lanczos_tridiagonal(
        model=model,
        criterion=criterion,
        batches=batches,
        params=params,
        steps=args.lanczos_steps,
    )
    eigvals = tridiagonal_eigvals(alphas, betas)
    eigvals = np.sort(eigvals)
    pos_eigvals = eigvals[eigvals > 1e-8]
    if pos_eigvals.size == 0:
        raise RuntimeError("Lanczos produced no positive eigenvalues.")

    eig_min = float(pos_eigvals.min())
    eig_max = float(pos_eigvals.max())
    span = max(eig_max - eig_min, 1e-6)
    bandwidth = max(span * args.bandwidth_ratio, 1e-3)
    xs = np.linspace(eig_min * 0.9, eig_max * 1.1, args.density_points)
    ys = kde_on_actual_axis(pos_eigvals, xs, bandwidth)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("eig")
    ax.set_ylabel("density")
    ax.set_title(f"Lanczos Hessian Eig-Density ({args.split})")
    ax.set_xlim(xs.min(), xs.max())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_prefix + "_density.png", dpi=220)
    plt.close(fig)

    payload = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "subset_size": args.subset_size,
        "max_batches": args.max_batches,
        "lanczos_steps": args.lanczos_steps,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "bandwidth": bandwidth,
        "lanczos_eigenvalues": pos_eigvals.tolist(),
    }
    with open(args.output_prefix + "_density.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
