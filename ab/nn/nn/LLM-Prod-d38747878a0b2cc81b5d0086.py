import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LPPoolBlock(nn.Module):
    """LPPool2d context branch + Hardsigmoid gate."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.LPPool2d(2, 4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        c = self.ctx(self.pool(h))
        c = nn.functional.interpolate(c, size=h.shape[-2:], mode="nearest")
        return self.conv2(h * self.gate(c)) + x



def supported_hyperparameters():
    return {'lr'}


class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        skip = self.skip(x)
        h = self.conv1(self.act(self.conv2(x)))
        return h + skip


class SPBlock(nn.Module):
    """Channel-preserving residual block (crossover contract)."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))) + x)


class LaplacianPyramidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.low_pass = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1)
        )
        self.high_pass = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        low = self.pool(x)
        high = x - self.up(low)
        low_denoised = self.low_pass(low)
        high_denoised = self.high_pass(high)
        return self.up(low_denoised) + high_denoised





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


class ELUAvgBlock(nn.Module):
    """ELU block with an AvgPool2d context branch."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AvgPool2d(4)
        self.ctx = nn.Conv2d(channels, channels, 1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode="nearest")
        return self.conv2(out + c) + x


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.initial_unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, 48, 3, padding=1)
        self.dp_blocks = nn.ModuleList([
            DualPathBlock(48),
            DualPathBlock(48),
            DualPathBlock(48)
        ])
        self.lp_block = LaplacianPyramidBlock(48)
        self.body = nn.Sequential(SPBlock(48), LPPoolBlock(48), DilBlock(48), ELUAvgBlock(48))
        self.out_conv = nn.Conv2d(48, 3 * 4, 3, padding=1)
        self.output_shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.initial_unshuffle(x)
        h = self.in_conv(h)
        for dp in self.dp_blocks:
            h = dp(h)
        h = self.lp_block(h)
        h = self.body(h)
        return torch.clamp(self.output_shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
