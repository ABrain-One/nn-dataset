import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'batch', 'epoch_max', 'transform'}

class DualPathBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.skip_conv(x)
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + skip + x)

class AdaptiveBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        g = self.pool(x)
        g = self.conv(g)
        g = self.gate(g)
        return x * g.expand_as(x)

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 3, padding=1)
        self.block1 = DualPathBlock(f, f)
        self.adapt1 = AdaptiveBlock(f)
        self.block2 = DualPathBlock(f, f)
        self.adapt2 = AdaptiveBlock(f)
        self.block3 = DualPathBlock(f, f)
        self.adapt3 = AdaptiveBlock(f)
        self.block4 = DualPathBlock(f, f)
        self.adapt4 = AdaptiveBlock(f)
        self.out_conv = nn.Conv2d(f, 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h = self.adapt1(self.block1(h))
        h = self.adapt2(self.block2(h))
        h = self.adapt3(self.block3(h))
        h = self.adapt4(self.block4(h))
        return torch.clamp(self.shuffle(self.out_conv(h)) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=prm.get("epoch_max", 50), eta_min=1e-5)

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
