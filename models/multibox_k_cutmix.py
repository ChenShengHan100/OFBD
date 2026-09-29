"""K-box CutMix helpers for OFBD sample-expansion semantics."""
import math

import torch
import torch.nn.functional as F

from models.paper_prior_rl import EnergyPriorPaperCutMixHelper
from models.rl_fg_cutmix import HeatmapCutMixHelper


class LocalEnergyMultiBoxHelper(HeatmapCutMixHelper):
    """Select K peak-jitter boxes and emit K independently mixed samples."""

    def build_cutmix(self, model, host_images, donor_images, target_a_onehot, target_b_onehot, lam):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, _, fg_mask = model(donor_images, return_aux=True)
        if was_training:
            model.train()

        h, w = host_images.shape[-2:]
        cut_ratio = math.sqrt(max(1.0 - lam, 0.0))
        cut_h, cut_w = max(int(h * cut_ratio), 1), max(int(w * cut_ratio), 1)
        candidates = self._peak_jitter_boxes(fg_mask, h, w, cut_h, cut_w)
        # K equals M here: every peak-jitter candidate becomes one training sample.
        chosen = candidates[:, : self.num_selected]
        batch = host_images.size(0)
        flat_boxes = chosen.reshape(batch * self.num_selected, 4)
        host = host_images.repeat_interleave(self.num_selected, dim=0)
        donor = donor_images.repeat_interleave(self.num_selected, dim=0)
        mixed, mask = self._build_mixed(host, donor, flat_boxes)
        actual_lam = 1.0 - mask.mean((1, 2, 3), keepdim=False)[:, None]
        ta = target_a_onehot.repeat_interleave(self.num_selected, dim=0)
        tb = target_b_onehot.repeat_interleave(self.num_selected, dim=0)
        return mixed.detach(), (actual_lam * ta + (1.0 - actual_lam) * tb).detach()


class EnergyPriorMultiBoxHelper(EnergyPriorPaperCutMixHelper):
    """Paper-prior RL selector that emits K independently mixed samples."""

    def build_cutmix(self, model, selector, selector_ref, host_images, donor_images,
                     target_a_onehot, target_b_onehot, target_a, target_b, lam,
                     patch_size, use_policy, entropy_weight, fg_reward_weight):
        del lam, patch_size, entropy_weight, fg_reward_weight
        was_training = model.training
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, donor_fmap, fg_mask = self._forward_aux(model, donor_images)
        if was_training:
            model.train()

        batch, _, image_h, image_w = host_images.shape
        boxes = self._random_boxes(batch, image_h, image_w, host_images.device)
        roi_features = self._roi_grid(donor_fmap.detach(), boxes, image_h, image_w)
        energy = self._roi_grid(fg_mask, boxes, image_h, image_w).mean((2, 3, 4))
        logits = selector(roi_features, energy)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        residual = logits[..., 1] - logits[..., 0]
        energy_scores = (energy - energy.mean(1, keepdim=True)) / energy.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        scores = self.energy_prior_weight * energy_scores + self.residual_weight * residual
        scores = torch.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        probs = F.softmax(scores, dim=1)
        with torch.no_grad():
            ref_logits = selector_ref(roi_features, energy)
            ref_logits = torch.nan_to_num(ref_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            ref_residual = ref_logits[..., 1] - ref_logits[..., 0]
            ref_residual = torch.nan_to_num(ref_residual, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            ref_scores = self.energy_prior_weight * energy_scores + self.residual_weight * ref_residual
            ref_scores = torch.nan_to_num(ref_scores, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            ref_probs = F.softmax(ref_scores, dim=1)
        probs = torch.nan_to_num(probs, nan=1.0 / probs.size(1), posinf=1.0, neginf=0.0)
        ref_probs = torch.nan_to_num(ref_probs, nan=1.0 / ref_probs.size(1), posinf=1.0, neginf=0.0)
        probs = probs.clamp_min(self.prob_eps)
        ref_probs = ref_probs.clamp_min(self.prob_eps)
        probs = probs / probs.sum(1, keepdim=True).clamp_min(self.prob_eps)
        ref_probs = ref_probs / ref_probs.sum(1, keepdim=True).clamp_min(self.prob_eps)

        if use_policy:
            chosen = torch.multinomial(probs, self.num_selected, replacement=False)
        else:
            chosen = energy_scores.topk(self.num_selected, dim=1).indices
        chosen_boxes = boxes.gather(1, chosen[..., None].expand(-1, -1, 4))
        host = host_images.repeat_interleave(self.num_selected, dim=0)
        donor = donor_images.repeat_interleave(self.num_selected, dim=0)
        mixed, mask = self._build_mixed(host, donor, chosen_boxes.reshape(batch * self.num_selected, 4))
        actual_lam = 1.0 - mask.mean((1, 2, 3), keepdim=False)[:, None]
        ta = target_a_onehot.repeat_interleave(self.num_selected, dim=0)
        tb = target_b_onehot.repeat_interleave(self.num_selected, dim=0)
        target = actual_lam * ta + (1.0 - actual_lam) * tb

        with torch.no_grad():
            _, reward_logits, _, _, _ = model(mixed)
            reward_logits = torch.nan_to_num(reward_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            reward_probs = F.softmax(reward_logits, dim=1)
            donor_labels = target_b.repeat_interleave(self.num_selected)
            host_labels = target_a.repeat_interleave(self.num_selected)
            reward = (reward_probs.gather(1, donor_labels[:, None]).squeeze(1) >
                      reward_probs.gather(1, host_labels[:, None]).squeeze(1)).float().view(batch, self.num_selected)

        current = probs.gather(1, chosen).clamp_min(self.prob_eps)
        reference = ref_probs.gather(1, chosen).clamp_min(self.prob_eps)
        advantage = (reward - reward.mean()).detach()
        ratio = current / reference
        ppo = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
        kl = (probs * (probs.clamp_min(self.prob_eps).log() - ref_probs.clamp_min(self.prob_eps).log())).sum(1).mean()
        selected_logits = logits.gather(1, chosen[..., None].expand(-1, -1, 2))
        aux_ce = F.cross_entropy(selected_logits.reshape(-1, 2), reward.long().reshape(-1))
        prior_ce = F.cross_entropy(residual, energy_scores.argmax(1))
        policy_loss = ppo + self.kl_weight * kl + aux_ce + 0.1 * prior_ce
        if not torch.isfinite(policy_loss):
            policy_loss = logits.sum() * 0.0
        return mixed.detach(), target.detach(), policy_loss, {
            'reward_mean': reward.mean().detach(), 'reward_std': reward.std(unbiased=False).detach(),
            'adv_abs_mean': advantage.abs().mean().detach(), 'kl_loss': kl.detach(), 'prior_ce': prior_ce.detach(),
        }
