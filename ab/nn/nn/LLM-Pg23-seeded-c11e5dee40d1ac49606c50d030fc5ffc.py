import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr'}


class DoubleDilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, channels * 2, 1)
        self.dw1 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.sca1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.proj_out1 = nn.Conv2d(channels, channels, 1)
        self.gamma1 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)
        
        self.dw2 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.sca2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.proj_out2 = nn.Conv2d(channels, channels, 1)
        self.gamma2 = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)

    def forward(self, inp):
        x = self.norm1(inp)
        xp = self.proj_in(x)
        x1, x2 = xp.chunk(2, dim=1)
        x1 = self.dw1(x1)
        x1 = x1 * x2
        x1 = x1 * self.sca1(x1)
        x1 = self.proj_out1(x1)
        x1 = x1 * self.gamma1
        
        x2 = self.dw2(x2)
        x2 = x2 * x1
        x2 = x2 * self.sca2(x2)
        x2 = self.proj_out2(x2)
        x2 = x2 * self.gamma2
        
        return inp + (x1 + x2)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = DoubleDilatedBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = Block(16)
        self.pool0 = nn.MaxPool2d(2)
        self.down0 = nn.Conv2d(16, 32, 3, padding=1)
        self.eb1 = Block(32)
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = nn.Conv2d(32, 64, 3, padding=1)
        self.eb2 = Block(64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bottleneck = Block(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.db2 = Block(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.db1 = Block(32)
        self.deconv0 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.db0 = Block(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e1 = self.down0(self.pool0(e0))
        e1 = self.eb1(e1)
        e2 = self.down1(self.pool1(e1))
        e2 = self.eb2(e2)
        bt = self.down2(self.pool2(e2))
        bt = self.bottleneck(bt)
        d2 = self.up2(torch.cat([self.deconv2(bt), e2], 1))
        d2 = self.db2(d2)
        d1 = self.up1(torch.cat([self.deconv1(d2), e1], 1))
        d1 = self.db1(d1)
        d0 = self.up0(torch.cat([self.deconv0(d1), e0], 1))
        d0 = self.db0(d0)
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
            noisy, clean = noisy.to(self.device), clean.to(self.device)
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
