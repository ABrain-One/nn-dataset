import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=in_shape[1], out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.spblock1 = SPBlock(32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.nlblock1 = NLBlock(64)
        self.spblock2 = SPBlock(64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.nlblock2 = NLBlock(128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.nlblock3 = NLBlock(64)
        self.spblock4 = SPBlock(64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.spblock5 = SPBlock(32)
        self.out_conv = nn.Conv2d(32, in_shape[1], kernel_size=3, padding=1)
        self.train_setup(prm)
        self.to(device)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.spblock1(x)
        x = self.down1(x)
        x = self.nlblock1(x)
        x = self.spblock2(x)
        x = self.down2(x)
        x = self.nlblock2(x)
        x = self.up2(x)
        x = self.nlblock3(x)
        x = self.spblock4(x)
        x = self.up1(x)
        x = self.spblock5(x)
        x = self.out_conv(x)
        x = torch.clamp(x + identity, 0.0, 1.0)
        return x

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = nn.MSELoss()(preds, clean)
            loss = loss_gt * 1000 + nn.L1Loss()(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)


class SPBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))) + x)


class NLBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        q = self.q(x)
        kp = F.pad(self.k(x), (3, 3, 3, 3))
        vp = F.pad(self.v(x), (3, 3, 3, 3))
        logits = []
        for dy in range(7):
            for dx in range(7):
                logits.append((q * kp[:, :, dy:dy + h, dx:dx + w]).sum(1) * self.scale)
        sim = torch.softmax(torch.stack(logits, dim=1), dim=1)
        out = x
        i = 0
        for dy in range(7):
            for dx in range(7):
                out = out + vp[:, :, dy:dy + h, dx:dx + w] * sim[:, i].unsqueeze(1)
                i += 1
        return out
