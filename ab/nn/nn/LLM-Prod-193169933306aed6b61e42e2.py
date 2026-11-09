import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def supported_hyperparameters():
    return {"lr"}

class MemoryGate(nn.Module):
    def __init__(self, channels, n_states):
        super().__init__()
        self.fuse = nn.Conv2d(channels * n_states, channels, 1)

    def forward(self, states):
        return self.fuse(torch.cat(states, dim=1))

class InstNormBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 1, padding=0)
        self.ctx_pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 5, padding=2)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.n_rec = 4
        self.head = nn.Conv2d(3, 32, 3, padding=1)
        self.rec = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1))
        self.act = nn.ReLU(inplace=True)
        self.gates = nn.ModuleList(
            [MemoryGate(32, i + 2) for i in range(self.n_rec)])
        self.tail = nn.Conv2d(32, 3, 3, padding=1)
        self.inst_norm = InstNormBlock(32)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.inst_norm(h)
        states = [h]
        for i in range(self.n_rec):
            r = self.act(self.rec(states[-1]) + states[-1])
            states.append(r)
            h = self.gates[i](states)
            states[-1] = h
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + \
                self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
