import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {"lr"}



class ScaleRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, img, prev):
        return img + self.net(torch.cat([img, prev], dim=1))

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.scale_refiner = ScaleRefiner(28)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        x4 = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        e4 = self.scale_refiner(x4, x4)
        e2 = self.scale_refiner(
            x2, F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False))
        e1 = self.scale_refiner(
            x, F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False))
        return torch.clamp(e1, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
