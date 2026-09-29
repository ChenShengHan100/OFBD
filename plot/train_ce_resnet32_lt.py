import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from plot.resnet32_cifar import resnet32_cifar


@dataclass
class EpochStats:
    loss: float
    top1: float
    many: float
    medium: float
    few: float


class LTSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.targets = [int(base_dataset.targets[i]) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone pure CE ResNet32 for CIFAR100-LT")
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--output-root", default="./log_sup")
    parser.add_argument("--file-name", default="standalone_pure_ce_resnet32")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--imb-factor", default=0.01, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--schedule", nargs="*", type=int, default=[160, 180])
    parser.add_argument("--cos", action="store_true")
    parser.add_argument("--print-freq", default=20, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=2058833602, type=int)
    parser.add_argument("--resume", default="", type=str)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def make_store_name(args: argparse.Namespace) -> str:
    return (
        f"{args.file_name}_{args.dataset}_resnet32_batchsize_{args.batch_size}"
        f"_epochs_{args.epochs}_lr_{args.lr}_wd_{args.weight_decay}"
        f"_if_{args.imb_factor}_seed_{args.seed}"
    )


def build_cifar100_lt_indices(targets: List[int], num_classes: int, imb_factor: float) -> Tuple[List[int], List[int]]:
    targets_np = np.array(targets, dtype=np.int64)
    img_max = len(targets) / num_classes
    img_num_per_cls = [
        int(img_max * (imb_factor ** (cls_idx / (num_classes - 1.0))))
        for cls_idx in range(num_classes)
    ]

    selected_indices: List[int] = []
    for cls_idx, cls_count in enumerate(img_num_per_cls):
        cls_indices = np.where(targets_np == cls_idx)[0]
        selected_indices.extend(cls_indices[:cls_count].tolist())
    return selected_indices, img_num_per_cls


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, List[int]]:
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_base = datasets.CIFAR100(
        root=args.data, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=args.data, train=False, download=True, transform=test_transform
    )
    selected_indices, cls_num_list = build_cifar100_lt_indices(
        train_base.targets, args.num_classes, args.imb_factor
    )
    train_dataset = LTSubset(train_base, selected_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, test_loader, cls_num_list


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return 100.0 * pred.eq(target).float().mean().item()


def shot_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    cls_num_list: List[int],
    many_thr: int = 100,
    few_thr: int = 20,
) -> Tuple[float, float, float]:
    class_correct = [0.0 for _ in cls_num_list]
    class_total = [0.0 for _ in cls_num_list]
    for pred, label in zip(preds.tolist(), labels.tolist()):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    many, medium, few = [], [], []
    for cls_idx, train_count in enumerate(cls_num_list):
        if class_total[cls_idx] == 0:
            continue
        acc = class_correct[cls_idx] / class_total[cls_idx]
        if train_count > many_thr:
            many.append(acc)
        elif train_count < few_thr:
            few.append(acc)
        else:
            medium.append(acc)

    def mean_or_zero(values: List[float]) -> float:
        return 100.0 * (sum(values) / len(values)) if values else 0.0

    return mean_or_zero(many), mean_or_zero(medium), mean_or_zero(few)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> float:
    if args.cos:
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        lr = args.lr
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> float:
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        running_top1 += accuracy(logits.detach(), targets) * batch_size

        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}]\t"
                f"Loss {loss.item():.4f}\t"
                f"Prec@1 {accuracy(logits.detach(), targets):.3f}\t"
                f"Time {time.time() - end:.3f}"
            )
        end = time.time()

    return running_loss / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cls_num_list: List[int],
    device: torch.device,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total += batch_size
        total_loss += loss.item() * batch_size
        total_top1 += accuracy(logits, targets) * batch_size
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(targets.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    many, medium, few = shot_metrics(preds, labels, cls_num_list)
    return EpochStats(
        loss=total_loss / total,
        top1=total_top1 / total,
        many=many,
        medium=medium,
        few=few,
    )


def save_checkpoint(
    state: Dict,
    is_best: bool,
    out_dir: str,
) -> None:
    latest_path = os.path.join(out_dir, "ckpt_latest.pth.tar")
    best_path = os.path.join(out_dir, "ckpt_best.pth.tar")
    torch.save(state, latest_path)
    if is_best:
        shutil.copyfile(latest_path, best_path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, cls_num_list = build_dataloaders(args)
    model = resnet32_cifar(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    store_name = make_store_name(args)
    out_dir = os.path.join(args.output_root, store_name)
    os.makedirs(out_dir, exist_ok=True)
    print(store_name)
    print(f"Use device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")

    best_top1 = 0.0
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint["best_top1"])
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        val_stats = validate(model, test_loader, criterion, cls_num_list, device)

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_stats.loss,
            "val_top1": val_stats.top1,
            "many_top1": val_stats.many,
            "medium_top1": val_stats.medium,
            "few_top1": val_stats.few,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: "
            f"val_top1={val_stats.top1:.3f}, "
            f"many={val_stats.many:.3f}, "
            f"med={val_stats.medium:.3f}, "
            f"few={val_stats.few:.3f}"
        )

        is_best = val_stats.top1 > best_top1
        best_top1 = max(best_top1, val_stats.top1)
        save_checkpoint(
            {
                "epoch": epoch,
                "arch": "resnet32_cifar",
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_top1": best_top1,
                "args": vars(args),
                "cls_num_list": cls_num_list,
            },
            is_best=is_best,
            out_dir=out_dir,
        )
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best Prec@1: {best_top1:.3f}")


if __name__ == "__main__":
    main()
