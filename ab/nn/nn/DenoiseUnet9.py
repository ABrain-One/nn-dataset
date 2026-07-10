import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2): smoother than L1, more robust than L2."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class _Norm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _GateActivation(nn.Module):
    """Split channels in half and multiply — activation-free non-linearity."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class _GatedAdaptiveBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c = channels

        self.norm1 = _Norm2d(c)
        self.proj_in = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.proj_out = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)

        self.norm2 = _Norm2d(c)
        self.ffn_up = nn.Conv2d(c, c * 4, 1)
        self.ffn_dn = nn.Conv2d(c * 2, c, 1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)

    def forward(self, inp):
        x = self.norm1(inp)
        xp = self.proj_in(x)
        x1, x2 = xp.chunk(2, dim=1)
        x1 = self.dw(x1)
        x = x1 * x2
        x = x * self.sca(x)
        x = self.proj_out(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.ffn_up(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.ffn_dn(x)
        return y + x * self.gamma


class GatedAdaptiveUNet(nn.Module):
    def __init__(self, in_ch=3, num_filters=32, enc_blocks=None, dec_blocks=None, mid_blocks=2):
        super().__init__()
        if enc_blocks is None:
            enc_blocks = [1, 1, 2, 2]
        if dec_blocks is None:
            dec_blocks = [1, 1, 1, 1]

        f = num_filters
        self.intro = nn.Conv2d(in_ch, f, 3, padding=1)

        self.enc_layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_channels = []

        for n in enc_blocks:
            self.enc_layers.append(nn.Sequential(*[_GatedAdaptiveBlock(f) for _ in range(n)]))
            enc_channels.append(f)
            self.downs.append(nn.Conv2d(f, f * 2, 2, stride=2))
            f *= 2

        self.mid = nn.Sequential(*[_GatedAdaptiveBlock(f) for _ in range(mid_blocks)])

        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.dec_layers = nn.ModuleList()

        for i, n in enumerate(dec_blocks):
            skip_ch = enc_channels[-(i + 1)]
            self.ups.append(nn.Sequential(
                nn.Conv2d(f, f * 2, 1),
                nn.PixelShuffle(2),
            ))
            f = f // 2
            self.fusions.append(nn.Conv2d(f + skip_ch, f, 1))
            self.dec_layers.append(nn.Sequential(*[_GatedAdaptiveBlock(f) for _ in range(n)]))

        self.output = nn.Conv2d(f, in_ch, 3, padding=1)

    def forward(self, inp):
        x = self.intro(inp)
        skips = []

        for enc, down in zip(self.enc_layers, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.mid(x)

        for up, fuse, dec, skip in zip(self.ups, self.fusions, self.dec_layers, reversed(skips)):
            x = up(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = dec(x)

        return inp + self.output(x)


class Net(nn.Module):
    """
    NAFNet-style gated adaptive U-Net using LayerNorm, simple channel-gate
    (multiplicative) non-linearities, depthwise convolutions, simplified
    channel attention and PixelShuffle upsampling. Trained with a
    Charbonnier loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = GatedAdaptiveUNet(
            in_ch=3, num_filters=32,
            enc_blocks=[1, 1, 2, 2], dec_blocks=[1, 1, 1, 1], mid_blocks=2,
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
