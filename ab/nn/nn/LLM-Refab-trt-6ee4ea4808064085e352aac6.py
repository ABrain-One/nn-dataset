import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class MemoryGate(nn.Module):
    def __init__(self, channels, n_states):
        super().__init__()
        self.fuse = nn.Conv2d(channels * n_states, channels, 1)

    def forward(self, states):
        return self.fuse(torch.cat(states, dim=1))


class KMixBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.basis = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])
        self.gate = nn.Conv2d(channels, 4, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=1)
        out = 0.0
        for i, conv in enumerate(self.basis):
            out += conv(x) * g[:, i:i + 1]
        return self.act(out + x)


class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.actv = nn.ELU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 32
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.rec = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1))
        self.act = nn.ReLU(inplace=True)
        self.gates = nn.ModuleList(
            [MemoryGate(f, i + 2) for i in range(4)])
        self.kmix = KMixBlock(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        states = [h]
        for i in range(4):
            r = self.act(self.rec(states[-1]) + states[-1])
            states.append(r)
            h = self.gates[i](states)
            states[-1] = h
        h = self.kmix(h)
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
