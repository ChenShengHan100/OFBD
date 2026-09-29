import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from plot.resnet32_cifar import resnet32_cifar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Hessian statistics for standalone pure CE ResNet32 checkpoints."
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
    parser.add_argument("--power-iters", default=20, type=int)
    parser.add_argument("--trace-samples", default=16, type=int)
    parser.add_argument("--gpu", default=7, type=int)
    parser.add_argument(
        "--output",
        default="./log_sup/standalone_ce_hessian_results.json",
    )
    return parser.parse_args()


def get_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = resnet32_cifar(num_classes=num_classes)
    return model.to(device)


def build_cls_num_list(checkpoint: Dict, num_classes: int, data_root: str) -> List[int]:
    if "cls_num_list" in checkpoint:
        return [int(x) for x in checkpoint["cls_num_list"]]

    train_base = datasets.CIFAR100(root=data_root, train=True, download=True)
    targets = train_base.targets
    imb_factor = float(checkpoint.get("args", {}).get("imb_factor", 0.01))
    img_max = len(targets) / num_classes
    return [
        int(img_max * (imb_factor ** (cls_idx / (num_classes - 1.0))))
        for cls_idx in range(num_classes)
    ]


def build_split_class_groups(
    cls_num_list: List[int],
    many_shot_thr: int = 100,
    low_shot_thr: int = 20,
) -> Dict[str, List[int]]:
    groups = {"all": list(range(len(cls_num_list))), "head": [], "medium": [], "tail": []}
    for class_idx, count in enumerate(cls_num_list):
        if count > many_shot_thr:
            groups["head"].append(class_idx)
        elif count < low_shot_thr:
            groups["tail"].append(class_idx)
        else:
            groups["medium"].append(class_idx)
    return groups


def build_eval_loader_for_classes(
    data_root: str,
    classes: List[int],
    batch_size: int,
    workers: int,
    subset_size: int,
) -> DataLoader:
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
    class_set = set(classes)
    indices = [idx for idx, target in enumerate(dataset.targets) if int(target) in class_set]
    if subset_size > 0:
        indices = indices[: min(subset_size, len(indices))]
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def iter_batches(
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        yield inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def collect_trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def dot_tensors(xs: List[torch.Tensor], ys: List[torch.Tensor]) -> torch.Tensor:
    return sum(torch.sum(x * y) for x, y in zip(xs, ys))


def l2_norm_tensors(xs: List[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(x * x) for x in xs))


def normalize_tensors(xs: List[torch.Tensor], eps: float = 1e-12) -> List[torch.Tensor]:
    denom = l2_norm_tensors(xs).clamp_min(eps)
    return [x / denom for x in xs]


def make_rademacher_like(params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0) for param in params]


def forward_loss(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs)
    return criterion(logits, targets)


def hessian_vector_product(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    params: List[torch.nn.Parameter],
    vector: List[torch.Tensor],
) -> List[torch.Tensor]:
    hvps = [torch.zeros_like(param) for param in params]
    for inputs, targets in batches:
        loss = forward_loss(model, criterion, inputs, targets)
        grads_raw = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        grads = [
            grad if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads_raw, params)
        ]
        grad_dot_vec = dot_tensors(grads, vector)
        batch_hvp_raw = torch.autograd.grad(
            grad_dot_vec,
            params,
            retain_graph=False,
            allow_unused=True,
        )
        batch_hvp = [
            hvp if hvp is not None else torch.zeros_like(param)
            for hvp, param in zip(batch_hvp_raw, params)
        ]
        hvps = [accum + cur.detach() for accum, cur in zip(hvps, batch_hvp)]
    scale = 1.0 / max(1, len(batches))
    return [hvp * scale for hvp in hvps]


def estimate_top_eigenvalue(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    params: List[torch.nn.Parameter],
    power_iters: int,
) -> float:
    vector = normalize_tensors(make_rademacher_like(params))
    eigenvalue = 0.0
    for _ in range(power_iters):
        hvp = hessian_vector_product(model, criterion, batches, params, vector)
        vector = normalize_tensors(hvp)
        hvp = hessian_vector_product(model, criterion, batches, params, vector)
        eigenvalue = float(dot_tensors(vector, hvp).item())
    return eigenvalue


def estimate_trace(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    params: List[torch.nn.Parameter],
    trace_samples: int,
) -> float:
    values: List[float] = []
    for _ in range(trace_samples):
        vector = make_rademacher_like(params)
        hvp = hessian_vector_product(model, criterion, batches, params, vector)
        values.append(float(dot_tensors(vector, hvp).item()))
    return sum(values) / max(1, len(values))


def evaluate_loss_and_acc(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[float, float]:
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
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def plot_hessian_summary(results: Dict[str, Dict[str, float]], output_prefix: str) -> None:
    splits = [split for split in ["all", "head", "medium", "tail"] if split in results]
    top_eigs = [results[split]["top_hessian_eigenvalue"] for split in splits]
    traces = [results[split]["hessian_trace_estimate"] for split in splits]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(splits, top_eigs, color="#1f77b4")
    axes[0].set_title("Top Hessian Eigenvalue")
    axes[0].set_ylabel("Eigenvalue")

    axes[1].bar(splits, traces, color="#ff7f0e")
    axes[1].set_title("Hessian Trace Estimate")
    axes[1].set_ylabel("Trace")

    fig.tight_layout()
    fig.savefig(output_prefix + "_hessian.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu)
    checkpoint = load_checkpoint(args.ckpt, device)
    ckpt_args = checkpoint.get("args", {})
    num_classes = int(ckpt_args.get("num_classes", 100))
    cls_num_list = build_cls_num_list(checkpoint, num_classes, args.data)
    class_groups = build_split_class_groups(cls_num_list)

    model = build_model(num_classes, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    criterion = nn.CrossEntropyLoss().to(device)
    params = collect_trainable_params(model)

    results: Dict[str, Dict[str, float]] = {}
    for split in ["all", "head", "medium", "tail"]:
        classes = class_groups[split]
        if not classes:
            continue
        loader = build_eval_loader_for_classes(
            data_root=args.data,
            classes=classes,
            batch_size=args.batch_size,
            workers=args.workers,
            subset_size=args.subset_size,
        )
        batches = list(iter_batches(loader, device, args.max_batches))
        if not batches:
            continue
        loss_value, top1 = evaluate_loss_and_acc(model, criterion, batches)
        top_eig = estimate_top_eigenvalue(
            model, criterion, batches, params, power_iters=args.power_iters
        )
        trace = estimate_trace(
            model, criterion, batches, params, trace_samples=args.trace_samples
        )
        results[split] = {
            "num_classes": len(classes),
            "num_examples": int(sum(batch[1].size(0) for batch in batches)),
            "loss": loss_value,
            "top1": top1,
            "top_hessian_eigenvalue": top_eig,
            "hessian_trace_estimate": trace,
        }

    payload = {
        "checkpoint": args.ckpt,
        "batch_size": args.batch_size,
        "subset_size": args.subset_size,
        "max_batches": args.max_batches,
        "power_iters": args.power_iters,
        "trace_samples": args.trace_samples,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    output_prefix = os.path.splitext(args.output)[0]
    plot_hessian_summary(results, output_prefix)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
