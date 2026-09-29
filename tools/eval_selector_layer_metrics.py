import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.cifar import IMBALANCECIFAR100
from models.multibox_k_cutmix import EnergyPriorMultiBoxHelper
from models.paper_prior_rl import FeatureProposalSelector
from models.resnet32 import OFBDModel32
from utils import shot_acc


LAYER_CHANNELS = {"layer1": 16, "layer2": 32, "layer3": 64}
LAYER_RESOLUTION = {"layer1": "32x32", "layer2": "16x16", "layer3": "8x8"}


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def auc_rank(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def parse_best_from_log(log_path):
    best = {"overall": None, "many": None, "med": None, "few": None}
    if not log_path.exists():
        return best
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Best Prec@1:" not in line:
            continue
        parts = line.replace(",", "").split()
        best["overall"] = float(parts[2])
        best["many"] = float(parts[5]) * 100.0
        best["med"] = float(parts[8]) * 100.0
        best["few"] = float(parts[11]) * 100.0
    return best


def build_val_loader(data_root, batch_size, workers):
    args = SimpleNamespace(Background_sampler="uniform", Foreground_sampler="balance")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_dataset = IMBALANCECIFAR100(
        root=data_root,
        args=args,
        transform=val_transform,
        train=False,
        imb_factor=1,
        download=True,
    )
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


def evaluate_checkpoint(ckpt_path, layer, data_root, batch_size, workers, max_batches, fg_quantile, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("paper_selector_arch") or "lite"
    energy_weight = float(ckpt.get("paper_rl_energy_weight", 10.0))
    residual_weight = float(ckpt.get("paper_rl_residual_weight", 1.0))

    model = OFBDModel32(name="resnet32", feat_dim=128, num_classes=100, use_norm=False).to(device)
    model.load_state_dict(strip_module_prefix(ckpt["state_dict"]), strict=True)
    model.eval()

    selector = FeatureProposalSelector(
        in_channels=LAYER_CHANNELS[layer],
        base_channels=32,
        arch=arch,
    ).to(device)
    selector.load_state_dict(ckpt["paper_selector_state_dict"], strict=True)
    selector.eval()

    helper = EnergyPriorMultiBoxHelper(
        top_m=4,
        num_selected=2,
        min_scale=0.15,
        max_scale=0.55,
        kl_weight=0.02,
        selector_feature_layer=layer,
        energy_prior_weight=energy_weight,
        residual_weight=residual_weight,
    )
    loader = build_val_loader(data_root, batch_size, workers)

    selector_probs = []
    residual_scores = []
    policy_scores = []
    labels = []
    total_logits = []
    total_labels = []

    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feat, logits, _, _, _, fmap, fg_mask = model(images, return_aux=True, return_aux_layer=layer)
            del feat
            total_logits.append(logits.detach().cpu())
            total_labels.append(target.detach().cpu())

            batch, _, image_h, image_w = images.shape
            boxes = helper._random_boxes(batch, image_h, image_w, images.device)
            roi_features = helper._roi_grid(fmap.detach(), boxes, image_h, image_w)
            energy = helper._roi_grid(fg_mask, boxes, image_h, image_w).mean((2, 3, 4))
            selector_logits = selector(roi_features, energy)
            selector_logits = torch.nan_to_num(selector_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            residual = selector_logits[..., 1] - selector_logits[..., 0]
            energy_z = (energy - energy.mean(1, keepdim=True)) / energy.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
            policy = energy_weight * energy_z + residual_weight * residual
            fg_prob = F.softmax(selector_logits, dim=-1)[..., 1]

            threshold = torch.quantile(energy, fg_quantile, dim=1, keepdim=True)
            label = energy >= threshold

            selector_probs.append(fg_prob.detach().cpu())
            residual_scores.append(residual.detach().cpu())
            policy_scores.append(policy.detach().cpu())
            labels.append(label.detach().cpu())

    selector_probs = torch.cat([x.reshape(-1) for x in selector_probs]).numpy()
    residual_scores = torch.cat([x.reshape(-1) for x in residual_scores]).numpy()
    policy_scores = torch.cat([x.reshape(-1) for x in policy_scores]).numpy()
    labels = torch.cat([x.reshape(-1) for x in labels]).numpy().astype(np.int64)
    total_logits = torch.cat(total_logits)
    total_labels = torch.cat(total_labels)
    preds = F.softmax(total_logits, dim=1).max(dim=1).indices.cuda()
    shot_labels = total_labels.cuda()
    train_loader = DataLoader(
        IMBALANCECIFAR100(
            root=data_root,
            args=SimpleNamespace(Background_sampler="uniform", Foreground_sampler="balance"),
            transform=[transforms.ToTensor(), transforms.ToTensor(), transforms.ToTensor()],
            train=True,
            imb_factor=0.01,
            download=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    many, med, few, _ = shot_acc(preds, shot_labels, train_loader, acc_per_cls=False)

    def summarize(score_array):
        fg = labels == 1
        bg = labels == 0
        return {
            "score_fg": float(score_array[fg].mean()),
            "score_bg": float(score_array[bg].mean()),
            "fg_auroc": auc_rank(score_array, labels),
        }

    pred_acc = float((preds.cpu() == total_labels).float().mean().item() * 100.0)
    return {
        "layer": layer,
        "resolution": LAYER_RESOLUTION[layer],
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt["epoch"]),
        "best_acc1": float(ckpt["best_acc1"]),
        "eval_overall": pred_acc,
        "eval_many": float(many * 100.0),
        "eval_med": float(med * 100.0),
        "eval_few": float(few * 100.0),
        "selector_prob": summarize(selector_probs),
        "residual_score": summarize(residual_scores),
        "policy_score": summarize(policy_scores),
        "num_regions": int(labels.size),
        "num_fg": int(labels.sum()),
        "num_bg": int((labels == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./logs/q2_layer_ablation_ofbd_local_if100")
    parser.add_argument("--data", default="./dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--fg-quantile", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)

    results = []
    for layer in ("layer1", "layer2", "layer3"):
        matches = sorted(root.glob(f"OFBD_RL_if100_{layer}_*/OFBD_ckpt.best.pth.tar"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one checkpoint for {layer}, found {len(matches)}")
        row = evaluate_checkpoint(
            matches[0],
            layer,
            args.data,
            args.batch_size,
            args.workers,
            args.max_batches,
            args.fg_quantile,
            device,
        )
        log_path = root / f"ofbd_{layer}.out"
        row["log_best"] = parse_best_from_log(log_path)
        results.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    out_path = Path(args.out) if args.out else root / "selector_layer_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer", "resolution", "prob_score_fg", "prob_score_bg", "prob_fg_auroc",
                "policy_score_fg", "policy_score_bg", "policy_fg_auroc",
                "overall", "few", "num_regions", "checkpoint",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({
                "layer": row["layer"],
                "resolution": row["resolution"],
                "prob_score_fg": row["selector_prob"]["score_fg"],
                "prob_score_bg": row["selector_prob"]["score_bg"],
                "prob_fg_auroc": row["selector_prob"]["fg_auroc"],
                "policy_score_fg": row["policy_score"]["score_fg"],
                "policy_score_bg": row["policy_score"]["score_bg"],
                "policy_fg_auroc": row["policy_score"]["fg_auroc"],
                "overall": row["log_best"]["overall"],
                "few": row["log_best"]["few"],
                "num_regions": row["num_regions"],
                "checkpoint": row["checkpoint"],
            })
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
