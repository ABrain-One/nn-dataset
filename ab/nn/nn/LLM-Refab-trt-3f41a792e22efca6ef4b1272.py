import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


class NoiseEstimator(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1), nn.Softplus())

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.to_scale = nn.Conv2d(3, channels, 1)
        self.to_shift = nn.Conv2d(3, channels, 1)

    def forward(self, feat, sigma):
        return feat * (1.0 + torch.tanh(self.to_scale(sigma))) + self.to_shift(sigma)


class HierarchicalFeatureAggregation(nn.Module):
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.unshuffle = nn.PixelUnshuffle(scale_factor)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        h = self.unshuffle(x)
        h = self.shuffle(h)
        return h


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.est = NoiseEstimator(12)
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.film1 = FiLM(f)
        self.film2 = FiLM(f)
        self.film3 = FiLM(f)
        self.b1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(f, f, 3, padding=1))
        self.b2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(f, f, 3, padding=1))
        self.b3 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(f, f, 3, padding=1))
        self.act = nn.ReLU(inplace=True)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self._sigma = None
        self.train_setup(prm)
        self.to(self.device)

    def forward(self, x):
        sigma = self.est(x)
        self._sigma = sigma
        h = self.head(x)
        h = self.act(self.b1(self.film1(h, sigma)) + h)
        h = self.act(self.b2(self.film2(h, sigma)) + h)
        h = self.act(self.b3(self.film3(h, sigma)) + h)
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if self._sigma is not None:
                with torch.no_grad():
                    target = (noisy - clean).abs()
                loss = loss + self.criterion_l1(self._sigma, target) * 20
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
