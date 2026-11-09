import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "batch", "epoch_max", "transform"}


class SplitAndMergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.split = nn.Identity()
        self.conv_a = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_b = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.merge = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x):
        a, b = self.split(x), self.split(x)
        a, b = self.conv_a(a), self.conv_b(b)
        return self.merge(torch.cat((a, b), dim=1))


class LiteDenoisingBlock(nn.Module):
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


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.actv(self.conv1(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.actv(self.conv1(self.upsample(x)))


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = f0 * 2, f0 * 4, f0 * 8, f0 * 16

        # Encoder
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = SplitAndMergeBlock(f0, f0)
        self.down0 = DownBlock(f0, f1)
        self.eb1 = SplitAndMergeBlock(f1, f1)
        self.down1 = DownBlock(f1, f2)
        self.eb2 = SplitAndMergeBlock(f2, f2)
        self.down2 = DownBlock(f2, f3)
        self.eb3 = SplitAndMergeBlock(f3, f3)
        self.down3 = DownBlock(f3, f4)
        self.bottleneck = SplitAndMergeBlock(f4, f4)

        # Decoder
        self.up3 = UpBlock(f4, f3)
        self.db3 = LiteDenoisingBlock(f3, f3)
        self.up2 = UpBlock(f3, f2)
        self.db2 = LiteDenoisingBlock(f2, f2)
        self.up1 = UpBlock(f2, f1)
        self.db1 = LiteDenoisingBlock(f1, f1)
        self.up0 = UpBlock(f1, f0)
        self.db0 = LiteDenoisingBlock(f0, f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=prm.get("epoch_max", 50), eta_min=1e-5
        )

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        e3 = self.eb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.db3(self.up3(b))
        d2 = self.db2(self.up2(d3))
        d1 = self.db1(self.up1(d2))
        d0 = self.db0(self.up0(d1))

        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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
                    m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
