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
        description="Plot eig-density using deflated top eigenvalues for standalone CE."
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
    parser.add_argument("--top-k", default=8, type=int)
    parser.add_argument("--power-iters", default=20, type=int)
    parser.add_argument("--density-points", default=500, type=int)
    parser.add_argument("--bandwidth-ratio", default=0.06, type=float)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument(
        "--output-prefix",
        default="./log_sup/standalone_ce_hessian_deflated_all",
    )
    return parser.parse_args()


def clone_tensors(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.detach().clone() for x in xs]


def hvp_deflated(
    model: nn.Module,
    criterion: nn.Module,
    batches,
    params: List[torch.nn.Parameter],
    vector: List[torch.Tensor],
    eigenpairs: List[Tuple[float, List[torch.Tensor]]],
) -> List[torch.Tensor]:
    hvp = hessian_vector_product(model, criterion, batches, params, vector)
    for eigval, eigvec in eigenpairs:
        coeff = dot_tensors(eigvec, vector)
        hvp = [h - eigval * coeff * v for h, v in zip(hvp, eigvec)]
    return hvp


def estimate_top_eigenpair_deflated(
    model: nn.Module,
    criterion: nn.Module,
    batches,
    params: List[torch.nn.Parameter],
    power_iters: int,
    eigenpairs: List[Tuple[float, List[torch.Tensor]]],
) -> Tuple[float, List[torch.Tensor]]:
    vector = normalize_tensors(make_rademacher_like(params))
    eigenvalue = 0.0
    for _ in range(power_iters):
        hvp = hvp_deflated(model, criterion, batches, params, vector, eigenpairs)
        vector = normalize_tensors(hvp)
        hvp = hvp_deflated(model, criterion, batches, params, vector, eigenpairs)
        eigenvalue = float(dot_tensors(vector, hvp).item())
    return eigenvalue, clone_tensors(vector)


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
        raise RuntimeError("No batches available for density estimation.")

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    params = collect_trainable_params(model)

    eigenpairs: List[Tuple[float, List[torch.Tensor]]] = []
    eigenvalues: List[float] = []
    for _ in range(args.top_k):
        eigval, eigvec = estimate_top_eigenpair_deflated(
            model=model,
            criterion=criterion,
            batches=batches,
            params=params,
            power_iters=args.power_iters,
            eigenpairs=eigenpairs,
        )
        if eigval <= 1e-6:
            break
        eigenpairs.append((eigval, eigvec))
        eigenvalues.append(eigval)

    eigvals = np.asarray(sorted(eigenvalues), dtype=np.float64)
    eig_min = float(eigvals.min())
    eig_max = float(eigvals.max())
    span = max(eig_max - eig_min, 1e-6)
    bandwidth = max(span * args.bandwidth_ratio, 1e-3)
    xs = np.linspace(eig_min * 0.9, eig_max * 1.1, args.density_points)
    ys = kde_on_actual_axis(eigvals, xs, bandwidth)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("eig")
    ax.set_ylabel("density")
    ax.set_title(f"Deflated Hessian Eig-Density ({args.split})")
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
        "top_k": args.top_k,
        "power_iters": args.power_iters,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "bandwidth": bandwidth,
        "deflated_top_eigenvalues": eigvals.tolist(),
    }
    with open(args.output_prefix + "_density.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
