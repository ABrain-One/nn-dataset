import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2): smooth approximation to L1."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class _DWSResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.act2 = nn.GELU()
        self.conv4 = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.9)

    def forward(self, x):
        net = self.act1(self.dw(self.conv1(x)))
        net = self.conv2(net)
        middle = x * self.alpha + net
        net = self.act2(self.conv3(middle))
        net = self.conv4(net)
        return middle + net


class AdaptiveMultiScaleNet(nn.Module):
    def __init__(self, enc_blocks, dec_blocks, bottom_layers, num_filters):
        super().__init__()
        f = num_filters

        self.init_conv = nn.Conv2d(3, f, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self._enc_channels = []

        for n_blks in enc_blocks:
            self.enc_blocks.append(nn.ModuleList([_DWSResBlock(f) for _ in range(n_blks)]))
            self._enc_channels.append(f)
            self.downs.append(nn.Conv2d(f, f * 2, 2, stride=2))
            f = f * 2

        self.bottom = nn.ModuleList([_DWSResBlock(f) for _ in range(bottom_layers)])

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i, n_blks in enumerate(dec_blocks):
            self.ups.append(nn.Conv2d(f, f * 2, 1))
            f = f // 2
            self.dec_blocks.append(nn.ModuleList([_DWSResBlock(f) for _ in range(n_blks)]))

        self.final_conv = nn.Conv2d(f, 3, 3, 1, 1)

    def forward(self, inp):
        x = self.init_conv(inp)

        skips = []
        for enc_group, down in zip(self.enc_blocks, self.downs):
            for blk in enc_group:
                x = blk(x)
            skips.append(x)
            x = down(x)

        for blk in self.bottom:
            x = blk(x)

        for up_conv, dec_group, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = self.ps(up_conv(x)) + skip
            for blk in dec_group:
                x = blk(x)

        return self.final_conv(x)


class Net(nn.Module):
    """
    Adaptive multi-scale denoising U-Net built from depthwise-separable
    residual blocks (1x1 -> depthwise 3x3 -> 1x1) with PixelShuffle
    upsampling and learnable residual scaling. Trained with a Charbonnier
    (smooth-L1) objective.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = AdaptiveMultiScaleNet(
            enc_blocks=[1, 1, 1, 1], dec_blocks=[1, 1, 1, 1],
            bottom_layers=2, num_filters=32,
        )
        self.criterion = CharbonnierLoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()

            preds = self(noisy)
            loss = self.criterion(preds, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
