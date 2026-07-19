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


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _IdentityBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_blk = _ConvBlock(channels, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv_blk(x)
        out = self.norm(self.conv(out))
        return self.act(out + x)


class _ResBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, num_blocks, downsample=True):
        layers = [_ConvBlock(in_ch, out_ch, stride=2 if downsample else 1)]
        for _ in range(num_blocks - 1):
            layers.append(_IdentityBlock(out_ch))
        super().__init__(*layers)


class _UpsampleConcatBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.conv = nn.ConvTranspose2d(out_ch + skip_ch, out_ch, 3, stride=1, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualDenoiseNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        c = in_channels

        self.enc0 = _ConvBlock(c, 64, stride=2)
        self.skip1 = _IdentityBlock(64)
        self.enc2 = _ResBlock(64, 128, 2)
        self.skip2 = _IdentityBlock(128)
        self.enc3 = _ResBlock(128, 256, 2)
        self.skip3 = _IdentityBlock(256)
        self.enc4 = _ResBlock(256, 512, 2)
        self.bottleneck = _IdentityBlock(512)

        self.up1 = _UpsampleConcatBlock(512, 256, 256)
        self.id1 = _IdentityBlock(256)
        self.up2 = _UpsampleConcatBlock(256, 128, 128)
        self.id2 = _IdentityBlock(128)
        self.up3 = _UpsampleConcatBlock(128, 64, 64)
        self.id3 = _IdentityBlock(64)
        self.up4 = _UpsampleConcatBlock(64, c, 64)
        self.id4 = _IdentityBlock(64)

        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, inp):
        x = self.enc0(inp)
        s1 = self.skip1(x)
        x = self.enc2(s1)
        s2 = self.skip2(x)
        x = self.enc3(s2)
        s3 = self.skip3(x)
        x = self.enc4(s3)
        x = self.bottleneck(x)

        x = self.id1(self.up1(x, s3))
        x = self.id2(self.up2(x, s2))
        x = self.id3(self.up3(x, s1))
        x = self.id4(self.up4(x, inp))

        return inp + self.final_conv(x)


class Net(nn.Module):
    """
    ResNet-18-style residual denoising encoder-decoder with InstanceNorm,
    GELU activations, identity residual blocks and transposed-conv
    upsampling with skip concatenation. Trained with a Charbonnier loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = ResidualDenoiseNet(in_channels=3)
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
