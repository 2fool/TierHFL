# resnet.py
# ------------------------------------------------------------
# Cleaned federated-friendly CNN building blocks:
# - No BatchNorm: use LayerNormCNN (GroupNorm(1, C)) for conv features
# - GELU activations to reduce "dead neurons" vs ReLU
# - Client model exposes: shared_base / personalized_path / local_classifier
# - Server model has proper downsample for residual when stride!=1 or C changes
# - Global classifier with mild MLP head; supports class_num alias
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Normalization helpers (BN-free)
# -----------------------------
class LayerNormCNN(nn.Module):
    """
    BN-free normalization for conv features.
    Implemented with GroupNorm(1, C), which behaves like InstanceNorm with affine.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, eps=eps, affine=True)

    def forward(self, x):
        return self.norm(x)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

# -----------------------------
# Residual BasicBlock (BN-free)
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.ln1   = LayerNormCNN(planes)
        self.act1  = nn.GELU()

        self.conv2 = conv3x3(planes, planes, 1)
        self.ln2   = LayerNormCNN(planes)

        self.downsample = downsample
        self.act_out    = nn.GELU()

        # note: weights are initialized in parent modules' _init_weights()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.ln2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act_out(out)
        return out

# -----------------------------
# Client-side model (Tier-aware)
# -----------------------------
class TierAwareClientModel(nn.Module):
    """
    Client model with:
      - shared_base       : shared feature trunk (kept in sync across clients)
      - personalized_path : local/private refinement branch
      - local_classifier  : head for local predictions
    Output of forward: local_logits, shared_features, personal_features

    Design target for CIFAR-style inputs (3x32x32):
      shared features channel = 32  (kept stable for server-side input)
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 100):
        super().__init__()

        # ---- Shared trunk: 3 -> 32 channels, keep spatial ~32x32
        self.shared_base = nn.Sequential(
            conv3x3(input_channels, 32, stride=1),
            LayerNormCNN(32),
            nn.GELU(),
            BasicBlock(32, 32, stride=1, downsample=None),
            BasicBlock(32, 32, stride=1, downsample=None),
        )

        # ---- Personalized branch: keep 32 channels; you can deepen if needed
        self.personalized_path = nn.Sequential(
            BasicBlock(32, 32, stride=1, downsample=None),
            BasicBlock(32, 32, stride=1, downsample=None),
        )

        # ---- Local classifier head (BN-free, mild MLP)
        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),                # 32
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Dropout(0.1),             # reduced from 0.3 to mitigate early collapse
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Some torch versions don't support gain('gelu'); fallback to 'relu'
                try:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='gelu')
                except ValueError:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # trunc_normal_ may not exist in very old torch; if so, normal_ is ok
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.shared_base(x)                # [B, 32, H, W]  (H,W ~32)
        personal = self.personalized_path(shared)   # [B, 32, H, W]
        local_logits = self.local_classifier(personal)
        return local_logits, shared, personal

# -----------------------------
# Server-side feature extractor
# -----------------------------
class EnhancedServerModel(nn.Module):
    """
    Server continues from client shared_features (B, 32, H, W)
    and ups the channels to 64 with stride=2 residual stage,
    then projects to feature_dim.

    NOTE: Proper downsample for residual path is handled via _make_layer.
    """
    def __init__(self, model_type: str = 'resnet56', feature_dim: int = 128,
                 input_channels: int = 3, **kwargs):
        super().__init__()
        # one stage: 32 -> 64, downsample spatial by 2
        self.stage = self._make_layer(BasicBlock, in_planes=32, out_planes=64, blocks=3, stride=2)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Linear(64, feature_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Same fallback for older torch
                try:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='gelu')
                except ValueError:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                LayerNormCNN(out_planes),
            )
        layers = [block(in_planes, out_planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes, 1, None))
        return nn.Sequential(*layers)

    def forward(self, shared_features):
        x = self.stage(shared_features)  # -> [B, 64, H/2, W/2]
        x = self.proj(x)                 # -> [B, feature_dim]
        return x

# -----------------------------
# Global classifier (server head)
# -----------------------------
class ImprovedGlobalClassifier(nn.Module):
    """
    Global head taking server feature vectors. Supports class_num alias.
    """
    def __init__(self, feature_dim: int = 128, num_classes: int = 100, class_num=None, **kwargs):
        super().__init__()
        if class_num is not None:
            num_classes = class_num
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
