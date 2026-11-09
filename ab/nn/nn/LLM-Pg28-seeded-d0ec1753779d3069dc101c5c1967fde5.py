import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}





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


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = in_shape[1]
        dim = 32

        self.intro = nn.Conv2d(channels, dim, 3, padding=1)
        self.enc0 = DilBlock(dim)
        self.enc1 = DilBlock(dim)
        self.enc2 = DilBlock(dim)
        self.enc3 = DilBlock(dim)
        self.down0 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.latent = nn.Sequential(DilBlock(dim), DilBlock(dim))
        self.up2 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.dec0 = DilBlock(dim)
        self.dec1 = DilBlock(dim)
        self.dec2 = DilBlock(dim)
        self.ending = nn.Conv2d(dim, channels, 3, padding=1)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        y = self.intro(x)
        s0 = self.enc0(y)
        y = self.down0(s0)
        s1 = self.enc1(y)
        y = self.down1(s1)
        s2 = self.enc2(y)
        y = self.down2(s2)
        y = self.latent(y)
        y = self.up2(y)
        y = self.dec0(self.up0(y))
        y = self.up1(y)
        y = self.dec1(self.up0(s2))
        y = self.up0(y)
        y = self.dec2(self.up0(s1))
        y = self.ending(y)
        return torch.clamp(y + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

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
