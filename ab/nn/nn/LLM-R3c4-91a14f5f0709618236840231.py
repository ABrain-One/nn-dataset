import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 1, padding=0)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)

class MultiStageAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.chan_att = nn.Conv2d(channels, channels, 1)
        self.spatial_att = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.spatial_att(torch.cat([avg, mx], dim=1)))
        chan_w = torch.sigmoid(self.chan_att(x.mean(dim=(2, 3), keepdim=True)))
        return x * att * chan_w.expand_as(x)

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = LiteDenoisingBlock(f0, f0)
        self.att0 = MultiStageAttention(f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = LiteDenoisingBlock(f1, f1)
        self.att1 = MultiStageAttention(f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = LiteDenoisingBlock(f2, f2)
        self.att2 = MultiStageAttention(f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb3 = LiteDenoisingBlock(f3, f3)
        self.att3 = MultiStageAttention(f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = LiteDenoisingBlock(f4, f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.db3 = LiteDenoisingBlock(f3, f3)
        self.att3_up = MultiStageAttention(f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = LiteDenoisingBlock(f2, f2)
        self.att2_up = MultiStageAttention(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = LiteDenoisingBlock(f1, f1)
        self.att1_up = MultiStageAttention(f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 3, padding=1)
        self.db0 = LiteDenoisingBlock(f0, f0)
        self.att0_up = MultiStageAttention(f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)
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
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e0 = self.att0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e1 = self.att1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        e2 = self.att2(e2)
        e3 = self.down2(e2)
        e3 = self.eb3(e3)
        e3 = self.att3(e3)
        b = self.bottleneck(self.down3(e3))
        d3 = self.db3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d3 = self.att3_up(d3)
        d2 = self.db2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d2 = self.att2_up(d2)
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d1 = self.att1_up(d1)
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        d0 = self.att0_up(d0)
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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