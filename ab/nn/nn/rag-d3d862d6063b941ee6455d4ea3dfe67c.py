# Auto-generated single-file for CrissCrossAttention
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmcv.cnn.bricks.scale.Scale ----
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

# ---- mmcv.ops.cc_attention.NEG_INF_DIAG ----
def NEG_INF_DIAG(n: int, device: torch.device) -> torch.Tensor:
    """Returns a diagonal matrix of size [n, n].

    The diagonal are all "-inf". This is for avoiding calculating the
    overlapped element in the Criss-Cross twice.
    """
    return torch.diag(torch.tensor(float('-inf')).to(device).repeat(n), 0)

# ---- CrissCrossAttention (target) ----
class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module.

    .. note::
        Before v1.3.13, we use a CUDA op. Since v1.3.13, we switch
        to a pure PyTorch and equivalent implementation. For more
        details, please refer to https://github.com/open-mmlab/mmcv/pull/1201.

        Speed comparison for one forward pass

        - Input size: [2,512,97,97]
        - Device: 1 NVIDIA GeForce RTX 2080 Ti

        +-----------------------+---------------+------------+---------------+
        |                       |PyTorch version|CUDA version|Relative speed |
        +=======================+===============+============+===============+
        |with torch.no_grad()   |0.00554402 s   |0.0299619 s |5.4x           |
        +-----------------------+---------------+------------+---------------+
        |no with torch.no_grad()|0.00562803 s   |0.0301349 s |5.4x           |
        +-----------------------+---------------+------------+---------------+

    Args:
        in_channels (int): Channels of the input feature map.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of Criss-Cross Attention.

        Args:
            x (torch.Tensor): Input feature with the shape of
                (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output of the layer, with the shape of
            (batch_size, in_channels, height, width)
        """
        B, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = torch.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(
            H, query.device)
        energy_H = energy_H.transpose(1, 2)
        energy_W = torch.einsum('bchw,bchj->bhwj', query, key)
        attn = F.softmax(
            torch.cat([energy_H, energy_W], dim=-1), dim=-1)  # [B,H,W,(H+W)]
        out = torch.einsum('bciw,bhwi->bchw', value, attn[..., :H])
        out += torch.einsum('bchj,bhwj->bchw', value, attn[..., H:])

        out = self.gamma(out) + x
        out = out.contiguous()

        return out

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels})'
        return s

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.criss_cross_attn = CrissCrossAttention(in_channels=16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(-1, -1, 8, 8)
        x = self.criss_cross_attn(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
