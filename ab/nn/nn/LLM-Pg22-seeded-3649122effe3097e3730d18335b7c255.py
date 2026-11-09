import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LiteDRRNBlock(nn.Module):
    def __init__(self, channels):
        super(LiteDRRNBlock, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.se = SEAttention(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pointwise(self.se(self.depthwise(x))))

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = LiteDRRNBlock(16)
        self.down0 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = LiteDRRNBlock(32)
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 1, stride=2, padding=0), nn.ReLU(inplace=True))
        self.eb2 = LiteDRRNBlock(64)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = LiteDRRNBlock(128)
        self.pre2 = nn.Conv2d(128, 512, 1)
        self.shuf2 = nn.PixelShuffle(2)
        self.up2 = nn.Conv2d(192, 64, 1, padding=0)
        self.db2 = LiteDRRNBlock(64)
        self.pre1 = nn.Conv2d(64, 256, 1)
        self.shuf1 = nn.PixelShuffle(2)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.db1 = LiteDRRNBlock(32)
        self.pre0 = nn.Conv2d(32, 128, 1)
        self.shuf0 = nn.PixelShuffle(2)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.db0 = LiteDRRNBlock(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        d2 = self.db2(self.up2(torch.cat([self.shuf2(self.pre2(bt)), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.shuf1(self.pre1(d2)), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.shuf0(self.pre0(d1)), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
