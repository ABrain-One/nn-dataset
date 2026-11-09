import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr", "batch", "epoch_max", "transform"}


class FiLM(nn.Module):
    """Per-block scale-and-shift derived from the noise map."""
    def __init__(self, channels):
        super().__init__()
        self.to_scale = nn.Conv2d(3, channels, 1)
        self.to_shift = nn.Conv2d(3, channels, 1)

    def forward(self, feat, sigma):
        return feat * (1.0 + torch.tanh(self.to_scale(sigma))) + self.to_shift(sigma)


class MemGate(nn.Module):
    """Fuses ALL previous recursion states."""
    def __init__(self, channels, n_states):
        super().__init__()
        self.fuse = nn.Conv2d(channels * n_states, channels, 1)

    def forward(self, states):
        return self.fuse(torch.cat(states, dim=1))


class Net(nn.Module):
    """Recursive FiLM-conditioned denoiser with dense memory."""
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.n_rec = 4
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.film1 = FiLM(f)
        self.film2 = FiLM(f)
        self.film3 = FiLM(f)
        self.rec_unit = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1))
        self.act = nn.ReLU(inplace=True)
        self.mem_gates = nn.ModuleList([MemGate(f, i + 2) for i in range(self.n_rec)])
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        states = [h]
        for i in range(self.n_rec):
            r = self.act(self.rec_unit(states[-1]) + states[-1])
            states.append(r)
            h = self.mem_gates[i](states)
            states[-1] = h
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)