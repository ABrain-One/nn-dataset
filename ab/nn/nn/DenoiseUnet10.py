import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class FrequencyAwareLoss(nn.Module):
    """
    L1 pixel loss + L1 on FFT magnitude spectrum + finite-difference edge loss.
    Encourages recovery of fine texture, edges, and high-frequency detail.
    """
    def __init__(self, freq_weight=0.05, edge_weight=0.1):
        super().__init__()
        self.freq_weight = freq_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        pixel_loss = F.l1_loss(pred, target)
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        tgt_freq = torch.fft.rfft2(target, norm='ortho')
        freq_loss = F.l1_loss(pred_freq.abs(), tgt_freq.abs())
        diff = pred - target
        edge_loss = (diff[:, :, :, 1:] - diff[:, :, :, :-1]).abs().mean() + \
                    (diff[:, :, 1:, :] - diff[:, :, :-1, :]).abs().mean()
        return pixel_loss + self.freq_weight * freq_loss + self.edge_weight * edge_loss


def _gn(channels):
    """Largest power-of-2 group count that evenly divides channels, capped at 32."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def _conv_gn_silu(in_ch, out_ch, kernel=3, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=kernel // 2, groups=groups, bias=False),
        nn.GroupNorm(_gn(out_ch), out_ch),
        nn.SiLU(),
    )


class _ChannelRecalibration(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class _ResidualCRBlock(nn.Module):
    """3x3 -> 3x3 residual block with channel recalibration, GroupNorm and SiLU."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            _conv_gn_silu(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_gn(channels), channels),
        )
        self.cr = _ChannelRecalibration(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.cr(self.block(x)) + x)


class _PyramidContextModule(nn.Module):
    """
    Parallel dilated convolutions at multiple rates capture multi-scale context
    without extra downsampling.
    """
    def __init__(self, channels, rates=(1, 2, 4, 6)):
        super().__init__()
        branch_ch = channels // len(rates)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.GroupNorm(_gn(branch_ch), branch_ch),
                nn.SiLU(),
            )
            for r in rates
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), channels, 1, bias=False),
            nn.GroupNorm(_gn(channels), channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.merge(torch.cat([b(x) for b in self.branches], dim=1))


class PyramidDenoiseNet(nn.Module):
    def __init__(self, in_ch=3, num_filters=32, depth=4, blocks_per_level=2):
        super().__init__()
        f = num_filters

        self.intro = _conv_gn_silu(in_ch, f)

        self.enc_layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_channels = []

        for _ in range(depth):
            self.enc_layers.append(
                nn.Sequential(*[_ResidualCRBlock(f) for _ in range(blocks_per_level)])
            )
            enc_channels.append(f)
            self.downs.append(_conv_gn_silu(f, f * 2, kernel=3, stride=2))
            f *= 2

        aspp_ch = (f // 4) * 4
        self.pre_aspp = nn.Identity() if f == aspp_ch else _conv_gn_silu(f, aspp_ch)
        self.bottleneck = nn.Sequential(
            _PyramidContextModule(aspp_ch),
            *[_ResidualCRBlock(aspp_ch) for _ in range(2)],
        )
        self.post_aspp = nn.Identity() if f == aspp_ch else _conv_gn_silu(aspp_ch, f)

        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.dec_layers = nn.ModuleList()

        for i in range(depth):
            skip_ch = enc_channels[-(i + 1)]
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                _conv_gn_silu(f, f // 2),
            ))
            f = f // 2
            self.fusions.append(nn.Sequential(
                nn.Conv2d(f + skip_ch, f, 1, bias=False),
                nn.GroupNorm(_gn(f), f),
                nn.SiLU(),
            ))
            self.dec_layers.append(
                nn.Sequential(*[_ResidualCRBlock(f) for _ in range(blocks_per_level)])
            )

        self.output = nn.Conv2d(f, in_ch, 1)

    def forward(self, inp):
        x = self.intro(inp)
        skips = []

        for enc, down in zip(self.enc_layers, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.post_aspp(self.bottleneck(self.pre_aspp(x)))

        for up, fuse, dec, skip in zip(
            self.ups, self.fusions, self.dec_layers, reversed(skips)
        ):
            x = up(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = dec(x)

        return inp + self.output(x)


class Net(nn.Module):
    """
    Pyramid-context denoising U-Net with GroupNorm/SiLU residual blocks,
    squeeze-and-excitation channel recalibration and a multi-rate dilated
    pyramid-context bottleneck. Trained with a frequency-aware (L1 + FFT +
    edge) loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = PyramidDenoiseNet(
            in_ch=3, num_filters=32, depth=4, blocks_per_level=2,
        )
        self.criterion = FrequencyAwareLoss()

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
