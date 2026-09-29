import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    estimate_trace,
    get_device,
    hessian_vector_product,
    iter_batches,
    load_checkpoint,
    l2_norm_tensors,
    make_rademacher_like,
    normalize_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot approximate Hessian spectrum for standalone CE ResNet32."
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
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--power-iters", default=20, type=int)
    parser.add_argument("--gpu", default=7, type=int)
    parser.add_argument(
        "--output",
        default="./log_sup/standalone_ce_hessian_spectrum_all.json",
    )
    return parser.parse_args()


def clone_tensors(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.detach().clone() for x in xs]


def sub_project(
    vector: List[torch.Tensor],
    basis: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    out = clone_tensors(vector)
    for basis_vec in basis:
        coeff = dot_tensors(out, basis_vec)
        out = [v - coeff * b for v, b in zip(out, basis_vec)]
    return out


def estimate_topk_eigenvalues(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    params: List[torch.nn.Parameter],
    top_k: int,
    power_iters: int,
) -> List[float]:
    basis: List[List[torch.Tensor]] = []
    eigenvalues: List[float] = []

    for _ in range(top_k):
        vector = normalize_tensors(make_rademacher_like(params))
        vector = sub_project(vector, basis)
        if l2_norm_tensors(vector).item() < 1e-10:
            vector = normalize_tensors(make_rademacher_like(params))

        for _ in range(power_iters):
            hvp = hessian_vector_product(model, criterion, batches, params, vector)
            hvp = sub_project(hvp, basis)
            if l2_norm_tensors(hvp).item() < 1e-10:
                break
            vector = normalize_tensors(hvp)

        hvp = hessian_vector_product(model, criterion, batches, params, vector)
        eig = float(dot_tensors(vector, hvp).item())
        basis.append(clone_tensors(vector))
        eigenvalues.append(eig)

    return eigenvalues


def plot_spectrum(eigenvalues: List[float], output_prefix: str, split: str) -> None:
    ranks = list(range(1, len(eigenvalues) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    markerline, stemlines, baseline = ax.stem(ranks, eigenvalues)
    plt.setp(markerline, color="#d62728", markersize=6)
    plt.setp(stemlines, color="#1f77b4", linewidth=1.5)
    plt.setp(baseline, color="black", linewidth=0.8)
    ax.set_title(f"Approximate Hessian Spectrum ({split})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_prefix + "_spectrum.png", dpi=220)
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
        raise RuntimeError("No batches available for spectrum estimation.")

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    criterion = nn.CrossEntropyLoss().to(device)
    params = collect_trainable_params(model)

    eigenvalues = estimate_topk_eigenvalues(
        model=model,
        criterion=criterion,
        batches=batches,
        params=params,
        top_k=args.top_k,
        power_iters=args.power_iters,
    )
    trace = estimate_trace(model, criterion, batches, params, trace_samples=16)

    payload: Dict[str, object] = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "subset_size": args.subset_size,
        "max_batches": args.max_batches,
        "top_k": args.top_k,
        "power_iters": args.power_iters,
        "topk_eigenvalues": eigenvalues,
        "trace_estimate": trace,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_spectrum(eigenvalues, os.path.splitext(args.output)[0], args.split)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
