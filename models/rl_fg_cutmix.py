import math

import torch
import torch.nn.functional as F


def unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


class HeatmapCutMixHelper:
    def __init__(
        self,
        top_m: int = 12,
        num_selected: int = 1,
        min_scale: float = 0.15,
        max_scale: float = 0.55,
        kl_weight: float = 0.02,
        prob_eps: float = 1e-6,
        jitter_ratio: float = 0.25,
    ):
        self.top_m = top_m
        if not 1 <= num_selected <= top_m:
            raise ValueError("num_selected must be between 1 and top_m")
        self.num_selected = num_selected
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.kl_weight = kl_weight
        self.prob_eps = prob_eps
        self.jitter_ratio = jitter_ratio

    def _peak_jitter_boxes(self, fg_mask: torch.Tensor, image_h: int, image_w: int, cut_h: int, cut_w: int) -> torch.Tensor:
        """Generate RL candidates around the local-energy maximum.

        Each candidate starts at the per-image argmax of ``fg_mask`` and receives
        an independent center jitter.  With ``top_m == 1`` this is the same
        peak-guided box construction used by the local-energy branch.
        """
        batch_size, _, fmap_h, fmap_w = fg_mask.shape
        device = fg_mask.device
        peak_indices = fg_mask.reshape(batch_size, -1).argmax(dim=1)
        max_jitter_x = int(cut_w * self.jitter_ratio)
        max_jitter_y = int(cut_h * self.jitter_ratio)
        center_x = ((peak_indices % fmap_w) * image_w // fmap_w).unsqueeze(1)
        center_y = ((peak_indices // fmap_w) * image_h // fmap_h).unsqueeze(1)
        shift_x = torch.randint(-max_jitter_x, max_jitter_x + 1, (batch_size, self.top_m), device=device) if max_jitter_x else torch.zeros(batch_size, self.top_m, device=device, dtype=torch.long)
        shift_y = torch.randint(-max_jitter_y, max_jitter_y + 1, (batch_size, self.top_m), device=device) if max_jitter_y else torch.zeros(batch_size, self.top_m, device=device, dtype=torch.long)
        # Candidate zero is exactly the local-energy baseline.
        shift_x[:, 0] = 0
        shift_y[:, 0] = 0
        cx = (center_x + shift_x).clamp(0, image_w)
        cy = (center_y + shift_y).clamp(0, image_h)
        x1 = (cx - cut_w // 2).clamp(0, image_w - 1)
        y1 = (cy - cut_h // 2).clamp(0, image_h - 1)
        x2 = (cx + (cut_w - cut_w // 2)).clamp(1, image_w)
        y2 = (cy + (cut_h - cut_h // 2)).clamp(1, image_h)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _build_mask_from_boxes(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        _, _, image_h, image_w = images.shape
        x = torch.arange(image_w, device=images.device).view(1, 1, image_w)
        y = torch.arange(image_h, device=images.device).view(1, image_h, 1)
        x1, y1, x2, y2 = (boxes[:, i].view(-1, 1, 1) for i in range(4))
        mask = ((x >= x1) & (x < x2) & (y >= y1) & (y < y2)).to(images.dtype)
        return mask.unsqueeze(1).expand(-1, images.size(1), -1, -1)

    def _extract_patches(self, images: torch.Tensor, boxes: torch.Tensor, patch_size: int) -> torch.Tensor:
        batch_size, num_boxes = boxes.shape[:2]
        image_h, image_w = images.shape[-2:]
        flat_boxes = boxes.reshape(-1, 4).to(images.dtype)
        x1, y1, x2, y2 = flat_boxes.unbind(dim=1)
        u = torch.linspace(0.0, 1.0, patch_size, device=images.device, dtype=images.dtype)
        grid_x = x1[:, None, None] + (x2 - x1)[:, None, None] * u[None, None, :]
        grid_y = y1[:, None, None] + (y2 - y1)[:, None, None] * u[None, :, None]
        grid = torch.stack((2.0 * grid_x.expand(-1, patch_size, -1) / max(image_w - 1, 1) - 1.0,
                            2.0 * grid_y.expand(-1, -1, patch_size) / max(image_h - 1, 1) - 1.0), dim=-1)
        crops = F.grid_sample(images.repeat_interleave(num_boxes, dim=0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        return crops.reshape(batch_size, num_boxes, images.size(1), patch_size, patch_size)

    def _build_mixed(self, host_images: torch.Tensor, donor_images: torch.Tensor, boxes: torch.Tensor):
        cutmix_mask = self._build_mask_from_boxes(host_images, boxes)
        cutmix_sample = host_images * (1.0 - cutmix_mask) + donor_images * cutmix_mask
        return cutmix_sample, cutmix_mask

    def _compute_selector_loss(self, selection_probs, ref_probs, chosen_idx, rewards, entropy_weight):
        chosen_probs = selection_probs.gather(1, chosen_idx).clamp_min(self.prob_eps)
        log_prob = chosen_probs.log()
        # Do not reduce per image: with one action this makes the advantage 0.
        advantage = (rewards - rewards.mean()).detach()
        policy_loss = -(advantage * log_prob.squeeze(1)).mean()
        kl_loss = (selection_probs * (selection_probs.clamp_min(self.prob_eps).log() - ref_probs.clamp_min(self.prob_eps).log())).sum(dim=1).mean()
        entropy = -(selection_probs * selection_probs.clamp_min(self.prob_eps).log()).sum(dim=1).mean()
        return policy_loss + self.kl_weight * kl_loss - entropy_weight * entropy, advantage, kl_loss, entropy

    def build_cutmix(
        self,
        model,
        selector,
        selector_ref,
        host_images: torch.Tensor,
        donor_images: torch.Tensor,
        target_a_onehot: torch.Tensor,
        target_b_onehot: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        lam: float,
        patch_size: int,
        use_policy: bool,
        entropy_weight: float,
        fg_reward_weight: float,
    ):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, _, fg_mask = model(donor_images, return_aux=True)
        if was_training:
            model.train()

        image_h, image_w = host_images.size(-2), host_images.size(-1)
        cut_rat = math.sqrt(max(1.0 - lam, 0.0))
        cut_w = max(int(image_w * cut_rat), 1)
        cut_h = max(int(image_h * cut_rat), 1)

        candidate_boxes = self._peak_jitter_boxes(fg_mask, image_h, image_w, cut_h, cut_w)
        donor_patches = self._extract_patches(donor_images, candidate_boxes, patch_size)
        selector_logits = selector(donor_patches)
        selection_scores = selector_logits[..., 1] - selector_logits[..., 0]
        selection_probs = F.softmax(selection_scores, dim=1)

        with torch.no_grad():
            ref_logits = selector_ref(donor_patches)
            ref_scores = ref_logits[..., 1] - ref_logits[..., 0]
            ref_probs = F.softmax(ref_scores, dim=1)

        if use_policy:
            chosen_idx = torch.multinomial(selection_probs, self.num_selected, replacement=False)
        else:
            chosen_idx = torch.zeros(host_images.size(0), self.num_selected, dtype=torch.long, device=host_images.device)
        gather_idx = chosen_idx.unsqueeze(-1).expand(-1, -1, 4)
        chosen_boxes = candidate_boxes.gather(1, gather_idx)

        batch_size = host_images.size(0)
        host_images = host_images.repeat_interleave(self.num_selected, dim=0)
        donor_images = donor_images.repeat_interleave(self.num_selected, dim=0)
        chosen_boxes = chosen_boxes.reshape(batch_size * self.num_selected, 4)
        cutmix_sample, cutmix_mask = self._build_mixed(host_images, donor_images, chosen_boxes)

        actual_lam = 1.0 - cutmix_mask.mean(dim=(1, 2, 3)).unsqueeze(1)
        target_a_onehot = target_a_onehot.repeat_interleave(self.num_selected, dim=0)
        target_b_onehot = target_b_onehot.repeat_interleave(self.num_selected, dim=0)
        target_cutmix = actual_lam * target_a_onehot + (1.0 - actual_lam) * target_b_onehot

        with torch.no_grad():
            _, reward_logits, _, _, _ = model(cutmix_sample)
            reward_probs = F.softmax(reward_logits, dim=1)
            target_b = target_b.repeat_interleave(self.num_selected, dim=0)
            target_a = target_a.repeat_interleave(self.num_selected, dim=0)
            host_prob = reward_probs.gather(1, target_a.unsqueeze(1)).squeeze(1).clamp_min(self.prob_eps)
            donor_prob = reward_probs.gather(1, target_b.unsqueeze(1)).squeeze(1).clamp_min(self.prob_eps)
            donor_ratio = 1.0 - actual_lam.squeeze(1)
            label_reward = (1.0 - donor_ratio) * host_prob.log() + donor_ratio * donor_prob.log()
            coverages = []
            fmap_h, fmap_w = fg_mask.shape[-2:]
            for flat_idx, (x1, y1, x2, y2) in enumerate(chosen_boxes.tolist()):
                mx1, mx2 = int(x1 * fmap_w / image_w), min(max(int(math.ceil(x2 * fmap_w / image_w)), 1), fmap_w)
                my1, my2 = int(y1 * fmap_h / image_h), min(max(int(math.ceil(y2 * fmap_h / image_h)), 1), fmap_h)
                coverages.append(fg_mask[flat_idx // self.num_selected, 0, my1:my2, mx1:mx2].mean())
            fg_coverage = torch.stack(coverages)
            rewards = (label_reward + fg_reward_weight * fg_coverage).reshape(batch_size, self.num_selected)

        selector_loss, advantage, kl_loss, entropy = self._compute_selector_loss(
            selection_probs, ref_probs, chosen_idx, rewards, entropy_weight
        )
        if not use_policy:
            selector_loss = selector_loss.detach() * 0.0
        stats = {
            "reward_mean": rewards.mean().detach(),
            "reward_std": rewards.std(unbiased=False).detach(),
            "adv_abs_mean": advantage.abs().mean().detach(),
            "kl_loss": kl_loss.detach(),
            "entropy": entropy.detach(),
        }
        return cutmix_sample.detach(), target_cutmix.detach(), selector_loss, stats
