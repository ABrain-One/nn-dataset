import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, out_channels, 3, padding=1)
        self.actv = nn.ELU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)

class DilBlock(nn.Module):
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

class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 3, padding=1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 5, padding=2)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.pool = nn.MaxPool2d(4)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        qh = self.pool(self.q(x)).flatten(2)
        kh = self.pool(self.k(x)).flatten(2)
        vh = self.pool(self.v(x)).flatten(2)
        att = self.soft(torch.matmul(qh.transpose(1, 2), kh) / c ** 0.5)
        out = torch.matmul(vh, att.transpose(1, 2))
        out = out.reshape(b, c, h // 4, w // 4)
        out = nn.functional.interpolate(out, size=(h, w), mode='nearest')
        return x + self.proj(out)

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = LiteDenoisingBlock(f0, f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = LiteDenoisingBlock(f1, f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = DilBlock(f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb3 = AttnBlock(f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = LiteDenoisingBlock(f4, f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.db3 = LiteDenoisingBlock(f3, f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = DilBlock(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = LiteDenoisingBlock(f1, f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 3, padding=1)
        self.db0 = LiteDenoisingBlock(f0, f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        e3 = self.eb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.db3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d2 = self.db2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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
