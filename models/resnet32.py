import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import torch.nn.init as init


class LT_SimAM(torch.nn.Module):
    """
    Long-Tailed SimAM (LT-SimAM)
    结合空间抑制理论与 L2 能量阶级保护的无参数 3D 注意力机制
    """
    def __init__(self, e_lambda=1e-4, max_floor=0.55):
        super(LT_SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.max_floor = max_floor

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. SimAM energy
        var = torch.var(x, dim=[2, 3], keepdim=True, unbiased=True)
        mu = x.mean(dim=[2, 3], keepdim=True)

        x_minus_mu_square = (x - mu).pow(2)
        y = x_minus_mu_square / (4 * (var + self.e_lambda)) + 0.5
        base_attn = self.activation(y)

        # 2. Long-tail aware dynamic floor
        x_flat = x.view(b, c, -1)
        chan_norm = torch.norm(x_flat, dim=-1, keepdim=True)  # [B, C, 1]
        energy_norm = chan_norm / (chan_norm.max(dim=1, keepdim=True)[0] + 1e-5)

        gate = 1.0 - energy_norm
        dynamic_floor = (gate * self.max_floor).view(b, c, 1, 1)

        # 3. Channel recalibration
        final_attn = dynamic_floor + (1.0 - dynamic_floor) * base_attn
        return x * final_attn


class BGE(nn.Module):
    """
    Background-Guided Estimator (BGE)
    基于低能量偏置的背景估计器：能量越低，越可能是背景。
    """
    def __init__(self, init_temp=0.1):
        super(BGE, self).__init__()
        self.temp = nn.Parameter(torch.tensor([init_temp]))

    def forward(self, fmap):
        B, C, H, W = fmap.shape
        fmap_flat = fmap.reshape(B, C, H * W).transpose(1, 2)  # [B, HW, C]

        energy_scores = torch.norm(fmap_flat, dim=-1)  # [B, HW]

        energy_mean = energy_scores.mean(dim=-1, keepdim=True)
        energy_var = torch.var(energy_scores, dim=-1, keepdim=True, unbiased=False)
        energy_std = torch.sqrt(torch.clamp(energy_var, min=0.0) + 1e-5)

        energy_norm = (energy_scores - energy_mean) / energy_std
        temp = torch.clamp(self.temp, min=1e-3)

        # 低能量 -> 背景概率更高
        bg_weights = torch.sigmoid((-energy_norm) / temp)
        bg_mask = bg_weights.view(B, 1, H, W)
        return bg_mask


class OFBDAggregationHead(nn.Module):
    """
    Background-Guided Aggregation Head
    先估计背景，再通过前景-背景对比得到更可靠的判别特征
    """
    def __init__(self, channels=64, max_floor=0.55, init_temp=0.1,
                 use_simam=True, use_bge=True, bg_scale=0.5):
        super(OFBDAggregationHead, self).__init__()
        self.use_simam = use_simam
        self.use_bge = use_bge
        self.bg_scale = bg_scale

        if self.use_simam:
            self.feature_calib = LT_SimAM(e_lambda=1e-4, max_floor=max_floor)
        if self.use_bge:
            self.background_estimator = BGE(init_temp=init_temp)

    def forward(self, fmap_raw):
        B, C, H, W = fmap_raw.shape

        # 1. 通道提纯
        if self.use_simam:
            fmap_refined = self.feature_calib(fmap_raw)
        else:
            fmap_refined = fmap_raw

        # 2. 背景估计
        if self.use_bge:
            bg_mask = self.background_estimator(fmap_refined)   # [B, 1, H, W]
        else:
            bg_mask = torch.zeros((B, 1, H, W), device=fmap_raw.device)

        bg_mask = torch.clamp(bg_mask, min=1e-6, max=1.0)
        fg_mask = torch.clamp(1.0 - bg_mask, min=1e-6, max=1.0)

        fmap_flat = fmap_refined.reshape(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        fg_weights = fg_mask.reshape(B, -1).unsqueeze(-1)              # [B, HW, 1]
        bg_weights = bg_mask.reshape(B, -1).unsqueeze(-1)              # [B, HW, 1]

        # 3. foreground / background 分别聚合
        fg_feat = (fmap_flat * fg_weights).sum(dim=1) / (fg_weights.sum(dim=1) + 1e-5)
        bg_feat = (fmap_flat * bg_weights).sum(dim=1) / (bg_weights.sum(dim=1) + 1e-5)

        # 4. background-guided readout
        pooled_feat = fg_feat - self.bg_scale * bg_feat

        return pooled_feat, fmap_refined, fg_mask, bg_mask, fg_feat, bg_feat


def resnet32(fft_k=32, sigma=8.0):
    return _ResNet(_BasicBlock, [5, 5, 5], fft_k=fft_k, sigma=sigma)


model_dict = {
    'resnet32': [resnet32, 512]
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, fft_k=32, sigma=8.0):
        super(_ResNet, self).__init__()
        self.in_planes = 16
        self.fft_k = fft_k
        self.sigma = sigma

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_layers(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(out)      # [B, 16, 32, 32]
        layer2 = self.layer2(layer1)   # [B, 32, 16, 16]
        layer3 = self.layer3(layer2)   # [B, 64, 8, 8]
        return {'layer1': layer1, 'layer2': layer2, 'layer3': layer3}

    def forward(self, x):
        return self.forward_layers(x)['layer3']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


class OFBDModel32(nn.Module):
    def __init__(self, num_classes=10, name='resnet32', head='mlp',
                 use_norm=True, feat_dim=128, bg_scale=0.5):
        super(OFBDModel32, self).__init__()
        self.use_norm = use_norm

        self.aggr_head = OFBDAggregationHead(
            channels=64,
            max_floor=0.55,
            init_temp=0.1,
            use_simam=True,
            use_bge=True,
            bg_scale=bg_scale
        )

        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()

        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(64, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported')

        if use_norm:
            self.fc = NormedLinear(64, num_classes)
            self.head_fc = nn.Sequential(
                nn.Linear(64, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            self.fc = nn.Linear(64, num_classes)
            self.head_fc = nn.Sequential(
                nn.Linear(64, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x, return_aux=False, return_aux_layer='layer3'):
        if return_aux:
            layer_features = self.encoder.forward_layers(x)
            if return_aux_layer not in layer_features:
                raise ValueError('Unsupported return_aux_layer: ' + str(return_aux_layer))
            fmap_raw = layer_features['layer3']
            selector_fmap = layer_features[return_aux_layer]
        else:
            fmap_raw = self.encoder(x)
            selector_fmap = fmap_raw

        pooled_feat, fmap_refined, fg_mask, bg_mask, fg_feat, bg_feat = self.aggr_head(fmap_raw)

        # 仅保留 foreground mask 给外部 foreground-guided cutmix 使用
        self.last_soft_weights = fg_mask.detach()

        feat_mlp = F.normalize(self.head(pooled_feat), dim=1)
        logits = self.fc(pooled_feat)

        if self.use_norm:
            centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)
            unfn_centers = self.head_fc(self.fc.weight.T)
        else:
            centers_logits = F.normalize(self.head_fc(self.fc.weight), dim=1)
            unfn_centers = self.head_fc(self.fc.weight)

        unfn_feat = self.head(pooled_feat)

        if return_aux:
            # 保留旧接口，避免训练代码联动大改
            return feat_mlp, logits, centers_logits, unfn_centers, unfn_feat, selector_fmap, fg_mask
        else:
            return feat_mlp, logits, centers_logits, unfn_centers, unfn_feat
