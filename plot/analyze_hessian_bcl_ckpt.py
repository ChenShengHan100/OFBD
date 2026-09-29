import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from plot.resnet32_cifar import BasicBlock, CIFARResNet


class NormedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.uniform_(self.weight, -1, 1)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class CIFARResNet32NormHead(CIFARResNet):
    def __init__(self, num_classes: int = 100):
        super().__init__(BasicBlock, [5, 5, 5], num_classes=num_classes)
        self.fc = NormedLinear(64, num_classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Hessian for BCL-style resnet32 checkpoint.")
    parser.add_argument(
        "--ckpt",
        default="./bcl_ckpt.best.pth.tar",
    )
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--subset-size", default=1024, type=int)
    parser.add_argument("--max-batches", default=4, type=int)
    parser.add_argument("--power-iters", default=20, type=int)
    parser.add_argument("--trace-samples", default=8, type=int)
    parser.add_argument("--gpu", default=2, type=int)
    parser.add_argument(
        "--output",
        default="./log_sup/bcl_ckpt_hessian.json",
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


def build_cls_num_list(num_classes: int, imb_factor: float = 0.01) -> List[int]:
    img_max = 50000 / num_classes
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


def hessian_vector_product(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    params: List[torch.nn.Parameter],
    vector: List[torch.Tensor],
) -> List[torch.Tensor]:
    hvps = [torch.zeros_like(param) for param in params]
    for inputs, targets in batches:
        logits = model(inputs)
        loss = criterion(logits, targets)
        grads_raw = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )
        grads = [
            grad if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads_raw, params)
        ]
        grad_dot_vec = dot_tensors(grads, vector)
        batch_hvp_raw = torch.autograd.grad(
            grad_dot_vec, params, retain_graph=False, allow_unused=True
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


@torch.no_grad()
def evaluate_loss_and_acc(
    model: nn.Module,
    criterion: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in batches:
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += int(targets.size(0))
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu)
    checkpoint = load_checkpoint(args.ckpt, device)

    model = CIFARResNet32NormHead(num_classes=100).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_unexpected_prefixes = ("head.", "head_center.")
    filtered_unexpected = [
        key for key in unexpected if not key.startswith(allowed_unexpected_prefixes)
    ]
    if missing or filtered_unexpected:
        raise RuntimeError(
            f"State dict mismatch. missing={missing[:10]} unexpected={filtered_unexpected[:10]}"
        )
    model.eval()

    cls_num_list = build_cls_num_list(num_classes=100, imb_factor=0.01)
    class_groups = build_split_class_groups(cls_num_list)
    criterion = nn.CrossEntropyLoss().to(device)
    params = collect_trainable_params(model)

    results = {}
    for split in ["all", "head", "medium", "tail"]:
        loader = build_eval_loader_for_classes(
            data_root=args.data,
            classes=class_groups[split],
            batch_size=args.batch_size,
            workers=args.workers,
            subset_size=args.subset_size,
        )
        batches = list(iter_batches(loader, device, args.max_batches))
        loss_value, top1 = evaluate_loss_and_acc(model, criterion, batches)
        top_eig = estimate_top_eigenvalue(model, criterion, batches, params, args.power_iters)
        trace = estimate_trace(model, criterion, batches, params, args.trace_samples)
        results[split] = {
            "num_examples": int(sum(batch[1].size(0) for batch in batches)),
            "loss": loss_value,
            "top1": top1,
            "top_hessian_eigenvalue": top_eig,
            "hessian_trace_estimate": trace,
        }

    payload = {
        "checkpoint": args.ckpt,
        "best_acc1": checkpoint.get("best_acc1"),
        "epoch": checkpoint.get("epoch"),
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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
