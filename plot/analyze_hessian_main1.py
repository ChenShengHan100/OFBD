import argparse
import os
import json
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.cifar import IMBALANCECIFAR100
from loss.contrastive import OFBDContrastiveLoss
from models.resnet32 import OFBDModel32
from utils import GaussianBlur, CIFAR10Policy


class DeviceAwareLogitAdjust(nn.Module):
    def __init__(self, cls_num_list: List[int], tau: float = 1.0):
        super().__init__()
        cls_num = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_prob = cls_num / cls_num.sum()
        m_list = tau * torch.log(cls_prob)
        self.register_buffer("m_list", m_list.view(1, -1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.m_list, targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Hessian statistics for OFBD checkpoints."
    )
    parser.add_argument(
        "--ckpt-a",
        default="./log_sup/_cifar100_resnet32_batchsize_256_epochs_200_temp_0.1_cutmix_prob_0.0_beta_0.6_ce_loss_type_logit_adjust_topk_30_scaling_factor_20_255_tau_0.9_lr_0.15_uncutout-sim_seed_2058833602/OFBD_ckpt.best.pth.tar",
    )
    parser.add_argument(
        "--ckpt-b",
        default="./log_sup/_cifar100_resnet32_batchsize_256_epochs_200_temp_0.1_cutmix_prob_0.5_beta_0.6_ce_loss_type_logit_adjust_topk_30_scaling_factor_20_255_tau_0.9_lr_0.15_uncutout-sim_seed_2058833602/OFBD_ckpt.best.pth.tar",
    )
    parser.add_argument(
        "--data",
        default="./dataset",
        help="Dataset root used by OFBD training",
    )
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--feat-dim", default=128, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--imb-factor", default=0.01, type=float)
    parser.add_argument("--tau", default=0.9, type=float)
    parser.add_argument("--alpha", default=2.0, type=float)
    parser.add_argument("--beta", default=0.6, type=float)
    parser.add_argument(
        "--ce-loss-type",
        default="logit_adjust",
        choices=["logit_adjust", "cross_entropy"],
    )
    parser.add_argument(
        "--loss-mode",
        default="eval_ce",
        choices=["eval_ce", "train_total"],
        help="eval_ce uses validation CE only; train_total uses OFBD alpha*CE + beta*SCL on train batches.",
    )
    parser.add_argument(
        "--subset-size",
        default=512,
        type=int,
        help="Number of examples used for Hessian estimation.",
    )
    parser.add_argument(
        "--max-batches",
        default=4,
        type=int,
        help="How many mini-batches to use from the subset.",
    )
    parser.add_argument(
        "--power-iters",
        default=20,
        type=int,
        help="Power iterations for the top Hessian eigenvalue estimate.",
    )
    parser.add_argument(
        "--trace-samples",
        default=16,
        type=int,
        help="Hutchinson samples for trace estimate.",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="CUDA device id. Use -1 for CPU.",
    )
    parser.add_argument(
        "--output",
        default="./log_sup/hessian_main1_results.json",
    )
    return parser.parse_args()


def get_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def build_model(num_classes: int, feat_dim: int, device: torch.device) -> OFBDModel32:
    model = OFBDModel32(
        name="resnet32",
        feat_dim=feat_dim,
        num_classes=num_classes,
        use_norm=False,
    )
    return model.to(device)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def load_checkpoint_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
    if "state_dict" not in checkpoint:
        raise KeyError(f"'state_dict' not found in checkpoint: {path}")
    return strip_module_prefix(checkpoint["state_dict"])


def build_dataloaders(
    data_root: str,
    batch_size: int,
    workers: int,
    subset_size: int,
    imb_factor: float,
) -> Tuple[IMBALANCECIFAR100, IMBALANCECIFAR100]:
    dataset_args = argparse.Namespace(
        Background_sampler="uniform",
        Foreground_sampler="balance",
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = IMBALANCECIFAR100(
        root=data_root,
        args=dataset_args,
        download=True,
        imb_factor=imb_factor,
        transform=transform,
        train=True,
    )
    val_dataset = IMBALANCECIFAR100(
        root=data_root,
        args=dataset_args,
        download=True,
        imb_factor=1,
        transform=transform,
        train=False,
    )
    return train_dataset, val_dataset


def build_train_transform_main1_cifar() -> List[transforms.Compose]:
    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    uncut_augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    return [
        transforms.Compose(uncut_augmentation_regular),
        transforms.Compose(augmentation_sim_cifar),
        transforms.Compose(augmentation_sim_cifar),
    ]


def build_train_loader_for_total_loss(
    data_root: str,
    batch_size: int,
    workers: int,
    subset_size: int,
    imb_factor: float,
) -> Tuple[IMBALANCECIFAR100, DataLoader]:
    dataset_args = argparse.Namespace(
        Background_sampler="uniform",
        Foreground_sampler="balance",
    )
    train_dataset = IMBALANCECIFAR100(
        root=data_root,
        args=dataset_args,
        download=True,
        imb_factor=imb_factor,
        transform=build_train_transform_main1_cifar(),
        train=True,
    )
    subset_size = min(subset_size, len(train_dataset))
    train_subset = Subset(train_dataset, list(range(subset_size)))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_dataset, train_loader


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
    val_dataset: IMBALANCECIFAR100,
    classes: List[int],
    batch_size: int,
    workers: int,
    subset_size: int,
) -> DataLoader:
    class_set = set(classes)
    indices = [idx for idx, target in enumerate(val_dataset.targets) if int(target) in class_set]
    if subset_size > 0:
        indices = indices[: min(subset_size, len(indices))]
    subset = Subset(val_dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def build_criterion(
    cls_num_list: List[int],
    tau: float,
    ce_loss_type: str,
    device: torch.device,
):
    if ce_loss_type == "cross_entropy":
        return torch.nn.CrossEntropyLoss().to(device)
    criterion = DeviceAwareLogitAdjust(cls_num_list, tau=tau)
    return criterion.to(device)


def build_scl_criterion(cls_num_list: List[int], temp: float, device: torch.device):
    return OFBDContrastiveLoss(cls_num_list, temp).to(device)


def iter_batches(
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        yield inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def collect_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
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
    model: torch.nn.Module,
    criterion,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    _, logits, _, _, _ = model(inputs)
    return criterion(logits, targets)


def forward_total_train_loss(
    model: torch.nn.Module,
    criterion_ce,
    criterion_scl,
    sample_a: List[torch.Tensor],
    target_a: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    batch_size = target_a.size(0)
    inputs = torch.cat([sample_a[0], sample_a[1], sample_a[2]], dim=0)
    feat_mlp, logits_all, centers, _, _ = model(inputs)
    logits1, _, _ = torch.split(logits_all, [batch_size, batch_size, batch_size], dim=0)
    _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
    features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
    centers = centers[: logits1.size(1)]
    ce_loss = criterion_ce(logits1, target_a)
    scl_loss = criterion_scl(centers, features, target_a)
    return alpha * ce_loss + beta * scl_loss


def hessian_vector_product(
    model: torch.nn.Module,
    criterion,
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
    model: torch.nn.Module,
    criterion,
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
    model: torch.nn.Module,
    criterion,
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
    model: torch.nn.Module,
    criterion,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in batches:
            _, logits, _, _, _ = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += int(targets.size(0))
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def iter_train_total_batches(
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Iterable[Tuple[List[torch.Tensor], torch.Tensor]]:
    for batch_idx, data in enumerate(loader):
        if batch_idx >= max_batches:
            break
        sample_a, _sample_b, target_a, _target_b = data
        sample_a = [tensor.to(device, non_blocking=True) for tensor in sample_a]
        target_a = target_a.to(device, non_blocking=True)
        yield sample_a, target_a


def analyze_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    train_dataset: IMBALANCECIFAR100,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    model = build_model(args.num_classes, args.feat_dim, device)
    state_dict = load_checkpoint_state(checkpoint_path, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading {checkpoint_path}: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {checkpoint_path}: {unexpected[:10]}")

    criterion = build_criterion(
        cls_num_list=train_dataset.cls_num_list,
        tau=args.tau,
        ce_loss_type=args.ce_loss_type,
        device=device,
    )
    params = collect_trainable_params(model)
    loss_value, top1 = evaluate_loss_and_acc(model, criterion, batches)
    top_eig = estimate_top_eigenvalue(
        model, criterion, batches, params, power_iters=args.power_iters
    )
    trace = estimate_trace(
        model, criterion, batches, params, trace_samples=args.trace_samples
    )
    return {
        "loss": loss_value,
        "top1": top1,
        "top_hessian_eigenvalue": top_eig,
        "hessian_trace_estimate": trace,
    }


def hessian_vector_product_train_total(
    model: torch.nn.Module,
    criterion_ce,
    criterion_scl,
    batches: List[Tuple[List[torch.Tensor], torch.Tensor]],
    params: List[torch.nn.Parameter],
    vector: List[torch.Tensor],
    alpha: float,
    beta: float,
) -> List[torch.Tensor]:
    hvps = [torch.zeros_like(param) for param in params]
    for sample_a, target_a in batches:
        loss = forward_total_train_loss(
            model, criterion_ce, criterion_scl, sample_a, target_a, alpha, beta
        )
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


def estimate_top_eigenvalue_train_total(
    model: torch.nn.Module,
    criterion_ce,
    criterion_scl,
    batches: List[Tuple[List[torch.Tensor], torch.Tensor]],
    params: List[torch.nn.Parameter],
    power_iters: int,
    alpha: float,
    beta: float,
) -> float:
    vector = normalize_tensors(make_rademacher_like(params))
    eigenvalue = 0.0
    for _ in range(power_iters):
        hvp = hessian_vector_product_train_total(
            model, criterion_ce, criterion_scl, batches, params, vector, alpha, beta
        )
        vector = normalize_tensors(hvp)
        hvp = hessian_vector_product_train_total(
            model, criterion_ce, criterion_scl, batches, params, vector, alpha, beta
        )
        eigenvalue = float(dot_tensors(vector, hvp).item())
    return eigenvalue


def estimate_trace_train_total(
    model: torch.nn.Module,
    criterion_ce,
    criterion_scl,
    batches: List[Tuple[List[torch.Tensor], torch.Tensor]],
    params: List[torch.nn.Parameter],
    trace_samples: int,
    alpha: float,
    beta: float,
) -> float:
    values: List[float] = []
    for _ in range(trace_samples):
        vector = make_rademacher_like(params)
        hvp = hessian_vector_product_train_total(
            model, criterion_ce, criterion_scl, batches, params, vector, alpha, beta
        )
        values.append(float(dot_tensors(vector, hvp).item()))
    return sum(values) / max(1, len(values))


def evaluate_total_train_loss_and_acc(
    model: torch.nn.Module,
    criterion_ce,
    criterion_scl,
    batches: List[Tuple[List[torch.Tensor], torch.Tensor]],
    alpha: float,
    beta: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for sample_a, target_a in batches:
            batch_size = target_a.size(0)
            inputs = torch.cat([sample_a[0], sample_a[1], sample_a[2]], dim=0)
            feat_mlp, logits_all, centers, _, _ = model(inputs)
            logits1, _, _ = torch.split(logits_all, [batch_size, batch_size, batch_size], dim=0)
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            centers = centers[: logits1.size(1)]
            ce_loss = criterion_ce(logits1, target_a)
            scl_loss = criterion_scl(centers, features, target_a)
            total_loss += float((alpha * ce_loss + beta * scl_loss).item()) * batch_size
            total_correct += int((logits1.argmax(dim=1) == target_a).sum().item())
            total_examples += int(batch_size)
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def analyze_checkpoint_train_total(
    checkpoint_path: str,
    device: torch.device,
    train_dataset: IMBALANCECIFAR100,
    batches: List[Tuple[List[torch.Tensor], torch.Tensor]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    model = build_model(args.num_classes, args.feat_dim, device)
    state_dict = load_checkpoint_state(checkpoint_path, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading {checkpoint_path}: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {checkpoint_path}: {unexpected[:10]}")
    criterion_ce = build_criterion(
        cls_num_list=train_dataset.cls_num_list,
        tau=args.tau,
        ce_loss_type=args.ce_loss_type,
        device=device,
    )
    criterion_scl = build_scl_criterion(train_dataset.cls_num_list, temp=0.1, device=device)
    params = collect_trainable_params(model)
    loss_value, top1 = evaluate_total_train_loss_and_acc(
        model, criterion_ce, criterion_scl, batches, args.alpha, args.beta
    )
    top_eig = estimate_top_eigenvalue_train_total(
        model,
        criterion_ce,
        criterion_scl,
        batches,
        params,
        args.power_iters,
        args.alpha,
        args.beta,
    )
    trace = estimate_trace_train_total(
        model,
        criterion_ce,
        criterion_scl,
        batches,
        params,
        args.trace_samples,
        args.alpha,
        args.beta,
    )
    return {
        "loss": loss_value,
        "top1": top1,
        "top_hessian_eigenvalue": top_eig,
        "hessian_trace_estimate": trace,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu)
    train_dataset, val_dataset = build_dataloaders(
        data_root=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        subset_size=args.subset_size,
        imb_factor=args.imb_factor,
    )
    class_groups = build_split_class_groups(train_dataset.cls_num_list)
    split_batches: Dict[str, List] = {}
    split_sizes: Dict[str, int] = {}
    if args.loss_mode == "eval_ce":
        for split_name, classes in class_groups.items():
            loader = build_eval_loader_for_classes(
                val_dataset=val_dataset,
                classes=classes,
                batch_size=args.batch_size,
                workers=args.workers,
                subset_size=args.subset_size,
            )
            batches = list(iter_batches(loader, device, args.max_batches))
            if not batches:
                continue
            split_batches[split_name] = batches
            split_sizes[split_name] = sum(targets.size(0) for _, targets in batches)
    else:
        train_total_dataset, train_total_loader = build_train_loader_for_total_loss(
            data_root=args.data,
            batch_size=args.batch_size,
            workers=args.workers,
            subset_size=args.subset_size,
            imb_factor=args.imb_factor,
        )
        train_targets = list(train_total_dataset.targets)
        for split_name, classes in class_groups.items():
            class_set = set(classes)
            selected_indices = [idx for idx, target in enumerate(train_targets) if int(target) in class_set]
            selected_indices = selected_indices[: min(args.subset_size, len(selected_indices))]
            subset = Subset(train_total_dataset, selected_indices)
            loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
            batches = list(iter_train_total_batches(loader, device, args.max_batches))
            if not batches:
                continue
            split_batches[split_name] = batches
            split_sizes[split_name] = sum(targets.size(0) for _, targets in batches)
        train_dataset = train_total_dataset

    if not split_batches:
        raise RuntimeError("No evaluation batches were loaded for any split.")

    results = {
        "meta": {
            "script": os.path.abspath(__file__),
            "device": str(device),
            "dataset": args.dataset,
            "subset_size": args.subset_size,
            "max_batches": args.max_batches,
            "batch_size": args.batch_size,
            "tau": args.tau,
            "ce_loss_type": args.ce_loss_type,
            "loss_mode": args.loss_mode,
            "imb_factor": args.imb_factor,
            "power_iters": args.power_iters,
            "trace_samples": args.trace_samples,
            "split_sizes": split_sizes,
            "class_groups": class_groups,
        },
        "checkpoint_a": {"path": args.ckpt_a, "stats": {}},
        "checkpoint_b": {"path": args.ckpt_b, "stats": {}},
    }
    for split_name, batches in split_batches.items():
        if args.loss_mode == "eval_ce":
            results["checkpoint_a"]["stats"][split_name] = analyze_checkpoint(
                args.ckpt_a, device, train_dataset, batches, args
            )
            results["checkpoint_b"]["stats"][split_name] = analyze_checkpoint(
                args.ckpt_b, device, train_dataset, batches, args
            )
        else:
            results["checkpoint_a"]["stats"][split_name] = analyze_checkpoint_train_total(
                args.ckpt_a, device, train_dataset, batches, args
            )
            results["checkpoint_b"]["stats"][split_name] = analyze_checkpoint_train_total(
                args.ckpt_b, device, train_dataset, batches, args
            )

    results["delta"] = {}
    for split_name in results["checkpoint_a"]["stats"].keys():
        results["delta"][split_name] = {
            key: results["checkpoint_b"]["stats"][split_name][key]
            - results["checkpoint_a"]["stats"][split_name][key]
            for key in results["checkpoint_a"]["stats"][split_name].keys()
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
