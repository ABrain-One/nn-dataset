import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.elu_avg = ELUAvgBlock(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.elu_avg(x)))

class ProgressiveDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.actv(self.conv(x))

class ProgressiveUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(self.conv(x))

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
        c = F.interpolate(c, size=out.shape[-2:], mode="nearest")
        return self.conv2(out + c) + x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, out_channels, 1)
        self.key = nn.Conv2d(in_channels, out_channels, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, c, h*w).permute(0, 2, 1)
        k = self.key(x).view(b, c, h*w)
        v = self.value(x).view(b, c, h*w)
        attn = torch.matmul(q, k) / (c ** 0.5)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).view(b, c, h, w)
        return out

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = HybridBlock(channels, f0)
        self.down0 = ProgressiveDownsampling(f0, f1)
        self.down1 = ProgressiveDownsampling(f1, f2)
        self.down2 = ProgressiveDownsampling(f2, f3)
        self.down3 = ProgressiveDownsampling(f3, f4)
        self.bottleneck = HybridBlock(f4, f4)
        self.attention = AttentionBlock(f4, f4)
        self.up3 = ProgressiveUpsampling(f4, f3)
        self.up2 = ProgressiveUpsampling(f3, f2)
        self.up1 = ProgressiveUpsampling(f2, f1)
        self.up0 = ProgressiveUpsampling(f1, f0)
        self.out_conv = HybridBlock(f0, channels)
        self.criterion = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e1 = self.down0(e0)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        b = self.bottleneck(self.down3(e3))
        b = self.attention(b)
        d3 = self.up3(b)
        d2 = self.up2(d3)
        d1 = self.up1(d2)
        d0 = self.up0(d1)
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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