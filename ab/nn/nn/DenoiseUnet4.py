import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class MSSSIMLoss(nn.Module):
    """
    Multi-scale SSIM approximation: L1 loss + per-scale structural similarity terms.
    Downsamples with AvgPool at each scale and accumulates L1 error.
    """
    def __init__(self, scales=3, base_weight=0.5):
        super().__init__()
        self.scales = scales
        self.base_weight = base_weight

    def forward(self, pred, target):
        total = F.l1_loss(pred, target) * self.base_weight
        scale_weight = (1.0 - self.base_weight) / self.scales
        p, t = pred, target
        for _ in range(self.scales):
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
            total = total + F.l1_loss(p, t) * scale_weight
        return total


def _reflect_conv(in_ch, out_ch, kernel=3, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(kernel // 2),
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride),
    )


class _ChanGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, 1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(avg))


class _SpatGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class _DualGateBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = _reflect_conv(filters, filters)
        self.conv2 = _reflect_conv(filters, filters)
        self.chan_gate = _ChanGate(filters)
        self.spat_gate = _SpatGate()
        self.act = nn.GELU()

    def forward(self, x):
        net = self.act(self.conv1(x))
        net = self.chan_gate(net)
        net2 = x + net
        net = self.act(self.conv2(net2))
        net = self.spat_gate(net)
        return net + net2


class _DecBlock(nn.Module):
    def __init__(self, net_in_ch, filters):
        super().__init__()
        self.conv = _reflect_conv(net_in_ch, filters)
        h = filters // 2
        self.sp1 = _reflect_conv(h, filters)
        self.sp2 = _reflect_conv(h, filters)
        self.act = nn.GELU()

    def forward(self, x):
        net = self.act(self.conv(x))
        h = net.shape[1] // 2
        sp1 = self.act(self.sp1(net[:, :h]))
        sp2 = self.act(self.sp2(net[:, h:]))
        return torch.cat([sp1, sp2], dim=1)


class DualAttentionNet(nn.Module):
    def __init__(self, enc_blocks, dec_blocks, bottom_layers, num_filters):
        super().__init__()
        f = num_filters
        in_ch = 3

        self.enc_blks = nn.ModuleList()
        self.downs = nn.ModuleList()
        skip_channels = []

        for n_blks in enc_blocks:
            group = nn.ModuleList()
            for k in range(n_blks):
                group.append(nn.Sequential(
                    _reflect_conv(in_ch if k == 0 else f, f),
                    nn.GELU(),
                ))
                in_ch = f
            self.enc_blks.append(group)
            skip_channels.append(f)
            self.downs.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(f, f * 2, 3, stride=2),
            ))
            f = f * 2
            in_ch = f

        self.bottom = nn.ModuleList([_DualGateBlock(f) for _ in range(bottom_layers)])

        self.up_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.dec_blks = nn.ModuleList()
        x_ch = f

        for i, n_blks in enumerate(dec_blocks):
            f = f // 2
            skip_ch = skip_channels[-(i + 1)]
            self.up_convs.append(nn.ConvTranspose2d(x_ch, f, 1, stride=2, output_padding=1))
            self.skip_convs.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(skip_ch, f, 3),
            ))
            x_ch = f
            group = nn.ModuleList()
            for k in range(n_blks):
                group.append(_DecBlock(x_ch, f))
                x_ch = f * 2
            self.dec_blks.append(group)

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(x_ch, 3, 3),
        )
        self.act = nn.GELU()

    def forward(self, inp):
        x = inp
        skips = []

        for enc_group, down in zip(self.enc_blks, self.downs):
            for blk in enc_group:
                x = blk(x)
            skips.append(x)
            x = down(x)

        for blk in self.bottom:
            x = blk(x)

        for up, skip_conv, dec_group, skip in zip(
            self.up_convs, self.skip_convs, self.dec_blks, reversed(skips)
        ):
            net = self.act(up(x))
            sc = self.act(skip_conv(skip))
            x = net + sc
            for blk in dec_group:
                x = blk(x)

        return inp + self.final(x)


class Net(nn.Module):
    """
    Dual-attention denoising U-Net combining channel and spatial gating in
    the bottleneck, reflection-padded convolutions and split-path decoder
    blocks. Trained with a multi-scale SSIM (L1 pyramid) loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = DualAttentionNet(
            enc_blocks=[1, 1, 1, 1], dec_blocks=[1, 1, 1, 1],
            bottom_layers=1, num_filters=32,
        )
        self.criterion = MSSSIMLoss()

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
