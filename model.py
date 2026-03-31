import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


# ---------------------------------------------------------------------------
# GeM Pooling
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)

    def extra_repr(self) -> str:
        return f"p={self.p.data.tolist()[0]:.4f}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ---------------------------------------------------------------------------
# Drop Path
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_(mask := torch.rand(shape, dtype=x.dtype, device=x.device) + keep) / keep
    return x * mask


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    rand_tensor = torch.floor(rand_tensor + keep_prob)
    return x / keep_prob * rand_tensor


# ---------------------------------------------------------------------------
# CBAM-augmented Bottleneck
# ---------------------------------------------------------------------------

class CBAMBottleneck(nn.Module):
    def __init__(self, block, drop_path_rate: float = 0.0):
        super().__init__()
        self.block = block
        channels = block.conv3.out_channels
        self.cbam = CBAMBlock(channels)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)
        out = self.block.conv3(out)
        out = self.block.bn3(out)
        out = self.cbam(out)
        out = drop_path(out, self.drop_path_rate, self.training)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out = out + identity
        out = self.block.relu(out)
        return out


def inject_cbam(layer: nn.Sequential, drop_path_rate: float) -> nn.Sequential:
    return nn.Sequential(*[CBAMBottleneck(b, drop_path_rate) for b in layer])


# ---------------------------------------------------------------------------
# Advanced ResNet with Multi-Scale GeM Pooling
# ---------------------------------------------------------------------------

class AdvancedResNet(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 100,
        drop_path_rate: float = 0.2,
        dropout: float = 0.4,
        proj_dim: int = 512,
    ):
        super().__init__()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = inject_cbam(backbone.layer3, drop_path_rate * 0.5)
        self.layer4 = inject_cbam(backbone.layer4, drop_path_rate)

        c2 = backbone.layer2[-1].conv3.out_channels
        c3 = backbone.layer3[-1].conv3.out_channels
        c4 = backbone.layer4[-1].conv3.out_channels

        self.gem2 = GeM(p=3.0)
        self.gem3 = GeM(p=3.0)
        self.gem4 = GeM(p=3.0)

        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)
        self.bn4 = nn.BatchNorm1d(c4)

        fused_dim = c2 + c3 + c4

        self.proj = nn.Sequential(
            nn.Linear(fused_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_out")
        nn.init.zeros_(self.classifier.bias)
        nn.init.kaiming_normal_(self.proj[0].weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        p2 = self.gem2(f2).flatten(1)
        p3 = self.gem3(f3).flatten(1)
        p4 = self.gem4(f4).flatten(1)

        p2 = self.bn2(p2)
        p3 = self.bn3(p3)
        p4 = self.bn4(p4)

        fused = torch.cat([p2, p3, p4], dim=1)
        out = self.proj(fused)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_resnet50_advanced(
    num_classes: int = 100,
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
    dropout: float = 0.4,
) -> AdvancedResNet:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)
    return AdvancedResNet(backbone, num_classes, drop_path_rate, dropout)


def build_resnet101_advanced(
    num_classes: int = 100,
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
    dropout: float = 0.4,
) -> AdvancedResNet:
    weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet101(weights=weights)
    return AdvancedResNet(backbone, num_classes, drop_path_rate, dropout)


def build_resnet152_advanced(
    num_classes: int = 100,
    pretrained: bool = True,
    drop_path_rate: float = 0.25,
    dropout: float = 0.4,
) -> AdvancedResNet:
    weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet152(weights=weights)
    return AdvancedResNet(backbone, num_classes, drop_path_rate, dropout)


MODEL_REGISTRY = {
    "resnet50": build_resnet50_advanced,
    "resnet101": build_resnet101_advanced,
    "resnet152": build_resnet152_advanced,
}


def build_model(name: str, **kwargs) -> AdvancedResNet:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
