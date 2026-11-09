import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr'}

class AdaptiveChannelReductionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels // 2, 1)
        self.conv_expand = nn.Conv2d(channels // 2, channels, 1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        reduced = self.conv_reduce(x)
        expanded = self.conv_expand(reduced)
        return x + self.actv(expanded)

class DualPathBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.path1 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(nn.Conv2d(channels, channels, 5, padding=2), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 5, padding=2), nn.ReLU(inplace=True))
        self.merge = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        return self.merge(torch.cat([p1, p2], dim=1))

class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.stem = nn.Sequential(nn.Conv2d(3, f, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(f, f, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.adaptive_reduction = AdaptiveChannelReductionBlock(f)
        self.dual_path_block = DualPathBlock(f)
        self.res_unit = ResidualUnit(f)
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(f, 24, 3, padding=1), nn.ReLU(inplace=True))
        self.local = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(48, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        g = self.up(self.dual_path_block(self.res_unit(self.adaptive_reduction(self.stem(x)))))
        l = self.local(x)
        return torch.clamp(self.tail(torch.cat([g, l], dim=1)) + x, 0.0, 1.0)

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
            loss = self.criterion_mse(preds, clean) * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
