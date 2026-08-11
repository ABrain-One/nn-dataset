import torch
import torch.nn as nn
import torch.optim as optim

class MBConvBlock(nn.Module):
    """Inverted-residual (MobileNetV2-style): expand 1x1 -> depthwise -> project 1x1."""

    def __init__(self, channels):
        super().__init__()
        hidden = channels * 2
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.act(self.expand(x))
        out = self.act(self.dw(out))
        return x + self.project(out)

class Net(nn.Module):
    """Depth-3 U-Net with concat skips and a global residual. The repeated conv block is
    channel-preserving and interchangeable (crossover contract), so this seed can recombine
    with the other seeds via block_crossover."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        c = 3
        f0 = 24
        f1, f2, f3 = (f0 * 2, f0 * 4, f0 * 8)
        Block = MBConvBlock
        self.in_conv = nn.Conv2d(c, f0, 3, padding=1)
        self.eb0 = Block(f0)
        self.down0 = nn.Conv2d(f0, f1, 5, stride=2, padding=2)
        self.eb1 = Block(f1)
        self.down1 = nn.Conv2d(f1, f2, 3, stride=2, padding=1)
        self.eb2 = Block(f2)
        self.down2 = nn.Conv2d(f2, f3, 3, stride=2, padding=1)
        self.bottleneck = Block(f3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = Block(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = Block(f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 3, padding=1)
        self.db0 = Block(f0)
        self.out_conv = nn.Conv2d(f0, c, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.db2(self.up2(torch.cat([self.upsample(b), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all((torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in bn_layers))
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)