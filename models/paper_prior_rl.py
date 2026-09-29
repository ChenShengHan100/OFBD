"""Paper-style global-proposal PPO selector with a local-energy prior."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, shufflenet_v2_x0_5

from models.rl_fg_cutmix import HeatmapCutMixHelper


class _TorchvisionROIEncoder(nn.Module):
    """Official lightweight torchvision backbones applied to ROI feature tensors."""
    def __init__(self, in_channels, arch):
        super().__init__()
        # The official image models take three channels.  This adapter only maps
        # the 64-channel feature tensor to their expected input interface.
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        if arch == 'shufflenetv2_x0_5':
            backbone = shufflenet_v2_x0_5(weights=None)
            self.features = nn.Sequential(
                backbone.conv1, backbone.maxpool, backbone.stage2,
                backbone.stage3, backbone.stage4, backbone.conv5,
            )
            self.out_dim = backbone.fc.in_features
        elif arch == 'mobilenetv3_small':
            backbone = mobilenet_v3_small(weights=None)
            self.features = backbone.features
            self.out_dim = backbone.classifier[0].in_features
        else:
            raise ValueError('Unsupported torchvision selector architecture: ' + arch)

    def forward(self, x):
        # Upsampling changes no proposal content; it only prevents the official
        # classifiers' pooling stages from collapsing a 4x4 ROI to an invalid size.
        x = F.interpolate(self.input_adapter(x), size=(32, 32), mode='bilinear', align_corners=False)
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class FeatureProposalSelector(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=64, base_channels=None, arch='lite'):
        super().__init__()
        if arch not in ('lite', 'shufflenetv2_x0_5', 'mobilenetv3_small'):
            raise ValueError('Unsupported selector architecture: ' + arch)
        if base_channels is not None and arch == 'lite':
            hidden_dim = base_channels * 2
        self.arch = arch
        if arch == 'lite':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feat_dim = hidden_dim
        else:
            self.encoder = _TorchvisionROIEncoder(in_channels, arch)
            feat_dim = self.encoder.out_dim
        self.head = nn.Sequential(nn.Linear(feat_dim + 1, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, 2))

    def forward(self, roi_features, energy_prior):
        batch, proposals = roi_features.shape[:2]
        feat = self.encoder(roi_features.flatten(0, 1)).flatten(1)
        energy = (energy_prior - energy_prior.mean(1, keepdim=True)) / energy_prior.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        logits = self.head(torch.cat((feat, energy.flatten()[:, None]), dim=1))
        return logits.view(batch, proposals, 2)


class EnergyPriorPaperCutMixHelper(HeatmapCutMixHelper):
    """Global proposals; RL learns a residual correction to local energy."""
    def __init__(self, *args, energy_prior_weight=1.0, residual_weight=1.0,
                 selector_feature_layer='layer3', **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_prior_weight = energy_prior_weight
        self.residual_weight = residual_weight
        self.selector_feature_layer = selector_feature_layer

    def _forward_aux(self, model, images):
        try:
            return model(images, return_aux=True, return_aux_layer=self.selector_feature_layer)
        except TypeError:
            if self.selector_feature_layer != 'layer3':
                raise
            return model(images, return_aux=True)
    @staticmethod
    def _roi_grid(features, boxes, image_h, image_w, out_size=4):
        batch, proposals = boxes.shape[:2]
        fmap_h, fmap_w = features.shape[-2:]
        flat = boxes.reshape(-1, 4).to(features.dtype)
        x1, y1, x2, y2 = flat.unbind(1)
        u = torch.linspace(0, 1, out_size, device=features.device, dtype=features.dtype)
        gx = x1[:, None, None] + (x2 - x1)[:, None, None] * u[None, None, :]
        gy = y1[:, None, None] + (y2 - y1)[:, None, None] * u[None, :, None]
        grid = torch.stack((2 * gx.expand(-1, out_size, -1) / max(image_w - 1, 1) - 1,
                            2 * gy.expand(-1, -1, out_size) / max(image_h - 1, 1) - 1), dim=-1)
        crops = F.grid_sample(features.repeat_interleave(proposals, 0), grid, align_corners=True, padding_mode='border')
        return crops.view(batch, proposals, *crops.shape[1:])

    def _random_boxes(self, batch, image_h, image_w, device):
        count = self.top_m
        scale = torch.empty(batch, count, device=device).uniform_(self.min_scale, self.max_scale)
        width = (scale * image_w).round().long().clamp(1, image_w)
        height = (scale * image_h).round().long().clamp(1, image_h)
        x1 = (torch.rand(batch, count, device=device) * (image_w - width + 1).float()).floor().long()
        y1 = (torch.rand(batch, count, device=device) * (image_h - height + 1).float()).floor().long()
        return torch.stack((x1, y1, x1 + width, y1 + height), dim=-1)

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
        residual_scores = logits[..., 1] - logits[..., 0]
        energy_scores = (energy - energy.mean(1, keepdim=True)) / energy.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        # The policy can correct, rather than replace, the local-energy prior.
        scores = self.energy_prior_weight * energy_scores + self.residual_weight * residual_scores
        probs = F.softmax(scores, dim=1)
        with torch.no_grad():
            ref_logits = selector_ref(roi_features, energy)
            ref_scores = self.energy_prior_weight * energy_scores + self.residual_weight * (ref_logits[..., 1] - ref_logits[..., 0])
            ref_probs = F.softmax(ref_scores, dim=1)
        energy_target = energy_scores.argmax(1)
        prior_ce = F.cross_entropy(residual_scores, energy_target)
        if use_policy:
            chosen = torch.multinomial(probs, 1)
        else:
            chosen = energy_target[:, None]
        chosen_boxes = boxes.gather(1, chosen[:, :, None].expand(-1, -1, 4)).squeeze(1)
        mixed, mask = self._build_mixed(host_images, donor_images, chosen_boxes)
        actual_lam = 1.0 - mask.mean((1, 2, 3), keepdim=False)[:, None]
        target = actual_lam * target_a_onehot + (1 - actual_lam) * target_b_onehot
        with torch.no_grad():
            _, reward_logits, _, _, _ = model(mixed)
            reward_probs = F.softmax(reward_logits, dim=1)
            fg_prob = reward_probs.gather(1, target_b[:, None]).squeeze(1)
            bg_prob = reward_probs.gather(1, target_a[:, None]).squeeze(1)
            reward = (fg_prob > bg_prob).float()
        current = probs.gather(1, chosen).squeeze(1).clamp_min(self.prob_eps)
        reference = ref_probs.gather(1, chosen).squeeze(1).clamp_min(self.prob_eps)
        advantage = (reward - reward.mean()).detach()
        ratio = current / reference
        ppo = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
        kl = (probs * (probs.clamp_min(self.prob_eps).log() - ref_probs.clamp_min(self.prob_eps).log())).sum(1).mean()
        selected_logits = logits.gather(1, chosen[:, :, None].expand(-1, -1, 2)).squeeze(1)
        aux_ce = F.cross_entropy(selected_logits, reward.long())
        policy_loss = ppo + self.kl_weight * kl + aux_ce + 0.1 * prior_ce
        if not use_policy:
            policy_loss = prior_ce
        stats = {'reward_mean': reward.mean().detach(), 'reward_std': reward.std(unbiased=False).detach(),
                 'adv_abs_mean': advantage.abs().mean().detach(), 'kl_loss': kl.detach(), 'prior_ce': prior_ce.detach()}
        return mixed.detach(), target.detach(), policy_loss, stats
