import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "batch", "epoch_max", "transform"}


class DilBlock(nn.Module):
    """Multi-dilation channel-preserving residual block (crossover contract)."""

    def __init__(self, channels):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.d1(x))
        h = self.act(self.d2(h))
        return self.act(self.d4(h) + x)


class LiteDenoisingBlock(nn.Module):
    """
    Lightweight denoising block with an internal channel-reduction
    bottleneck (f -> f/2 -> f), standard 3x3 convolutions, hardware-native ReLU activations, 
    and a local residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


class EluCtxBlock(nn.Module):
    """ELU block with a weighted AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        w = F.softmax(c.view(c.size(0), -1), dim=-1).view_as(c)
        c = c * w
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x


class Net(nn.Module):
    """Hybrid Denoiser combining Multi-Dilation Blocks, Lightweight Denoising Blocks, and ELUCtxBlocks."""

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        channels = 32
        self.in_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.down0 = nn.MaxPool2d(2)
        self.dil_block0 = DilBlock(channels)
        self.ld_block0 = LiteDenoisingBlock(channels, channels)
        self.elu_ctx0 = EluCtxBlock(channels)
        self.down1 = nn.MaxPool2d(2)
        self.dil_block1 = DilBlock(channels)
        self.ld_block1 = LiteDenoisingBlock(channels, channels)
        self.elu_ctx1 = EluCtxBlock(channels)
        self.dil_block2 = DilBlock(channels)
        self.up1 = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.out_conv = nn.Conv2d(channels, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(x)
        h = self.down0(h)
        h = self.dil_block0(h)
        h = self.ld_block0(h)
        h = self.elu_ctx0(h)
        h = self.down1(h)
        h = self.dil_block1(h)
        h = self.ld_block1(h)
        h = self.elu_ctx1(h)
        h = self.dil_block2(h)
        h = self.up1(h)
        h = self.up2(h)
        return torch.clamp(self.out_conv(h) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 0.0001)
        batch = prm.get("batch", 16)
        epoch_max = prm.get("epoch_max", 50)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch_max, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
