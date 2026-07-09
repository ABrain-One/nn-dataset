import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class SmoothL1GradLoss(nn.Module):
    """Smooth-L1 pixel loss + finite-difference gradient loss for edge sharpness."""
    def __init__(self, grad_weight=0.15):
        super().__init__()
        self.grad_weight = grad_weight

    def _grad_loss(self, pred, target):
        diff = pred - target
        dx = diff[:, :, :, 1:] - diff[:, :, :, :-1]
        dy = diff[:, :, 1:, :] - diff[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target) + self.grad_weight * self._grad_loss(pred, target)


class _EncFeatBlock(nn.Module):
    def __init__(self, in_ch, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, filters // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(filters // 2, filters * 2, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))))


class _DecFeatBlock(nn.Module):
    def __init__(self, in_ch, skip_in_ch, filters):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_in_ch, filters, 3, padding=1)
        self.up = nn.ConvTranspose2d(in_ch, filters, 2, stride=2)
        self.conv = nn.Conv2d(filters, filters, 3, padding=1)
        self.act = nn.GELU()
        h = filters // 2
        self.sp1 = nn.Conv2d(h, h, 3, padding=1)
        self.sp2 = nn.Conv2d(h, h, 3, padding=1)

    def forward(self, x, skip):
        net = self.act(self.up(x)) + self.skip_conv(skip)
        net = self.act(self.conv(net))
        h = net.shape[1] // 2
        sp1 = self.act(self.sp1(net[:, :h]))
        sp2 = self.sp2(net[:, h:])
        return torch.cat([sp1, sp2], dim=1)


class HierarchFusionNet(nn.Module):
    def __init__(self, enc_blocks, dec_blocks, num_filters):
        super().__init__()
        f = num_filters

        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, f, 3, padding=1),
            nn.GELU(),
        )

        self.downs = nn.ModuleList()
        self.enc_blks = nn.ModuleList()
        skip_channels = []
        in_ch = f

        for n_blks in enc_blocks:
            skip_channels.append(in_ch)
            self.downs.append(nn.Conv2d(in_ch, f, 4, stride=2, padding=1))
            blk_in = f
            blk_group = nn.ModuleList()
            for _ in range(n_blks):
                blk_group.append(_EncFeatBlock(blk_in, f))
                blk_in = f * 2
            self.enc_blks.append(blk_group)
            in_ch = f * 2
            f = f * 2

        self.dec_blks = nn.ModuleList()
        f_dec = f // 2
        x_ch = in_ch
        for i, n_blks in enumerate(dec_blocks):
            f_dec = f_dec // 2
            skip_ch = skip_channels[-(i + 1)]
            blk_group = nn.ModuleList()
            for _ in range(n_blks):
                blk_group.append(_DecFeatBlock(x_ch, skip_ch, f_dec))
                x_ch = f_dec
                skip_ch = f_dec
            self.dec_blks.append(blk_group)
            x_ch = f_dec

        self.final_conv = nn.Conv2d(x_ch, 3, 3, padding=1)
        self._skip_channels = skip_channels

    def forward(self, inp):
        x = self.stem(inp)

        skips = []
        for down, enc_group in zip(self.downs, self.enc_blks):
            skips.append(x)
            x = down(x)
            for blk in enc_group:
                x = blk(x)

        for dec_group, skip in zip(self.dec_blks, reversed(skips)):
            for blk in dec_group:
                x = blk(x, skip)

        return inp + self.final_conv(x)


class Net(nn.Module):
    """
    Hierarchical fusion U-Net with channel-expanding encoder feature blocks
    and split-path decoder blocks using transposed-conv upsampling and a
    global residual connection. Trained with a smooth-L1 + gradient loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = HierarchFusionNet(
            enc_blocks=[1, 1, 1, 1], dec_blocks=[1, 1, 1, 1], num_filters=32,
        )
        self.criterion = SmoothL1GradLoss()

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
