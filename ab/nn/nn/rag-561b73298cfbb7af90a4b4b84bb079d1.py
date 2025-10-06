# Auto-generated single-file for FactorAttnConvRelPosEnc
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Tuple

# ---- original imports from contributing modules ----

# ---- FactorAttnConvRelPosEnc (target) ----
class FactorAttnConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, h, N, Ch]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # [B, h, N, Ch]

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

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
        self.crpe = self.build_crpe()
        self.factor_attn = FactorAttnConvRelPosEnc(dim=32, num_heads=4, qkv_bias=False, attn_drop=0.1, proj_drop=0.1, shared_crpe=self.crpe)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def build_crpe(self):
        class SimpleCRPE:
            def __call__(self, q, v, size):
                B, h, N, Ch = q.shape
                H, W = size
                q_reshaped = q.transpose(1, 2).reshape(B, N, h * Ch)
                v_reshaped = v.transpose(1, 2).reshape(B, N, h * Ch)
                return torch.zeros_like(q_reshaped).reshape(B, N, h, Ch).transpose(1, 2)
        return SimpleCRPE()

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.flatten(2).transpose(1, 2)
        x = self.factor_attn(x, size=(4, 4))
        x = x.mean(dim=1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
