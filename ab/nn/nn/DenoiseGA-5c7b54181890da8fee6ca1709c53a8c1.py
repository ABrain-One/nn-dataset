import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class CouplingBlock(nn.Module):
    """Additive coupling step (InvDN / normalizing-flow family): the channel axis is split in
    half, one half is transformed CONDITIONED ON the other, then the roles swap. Information
    crosses between halves only through the coupling transforms, and every step is exactly
    invertible by construction."""

    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        def t():
            return nn.Sequential(
                nn.Conv2d(half, half, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(half, half, 3, padding=1))
        self.t1 = t()
        self.t2 = t()

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        b = b + self.t1(a)          # b conditioned on a
        a = a + self.t2(b)          # a conditioned on the UPDATED b
        return torch.cat([a, b], dim=1)


class Net(nn.Module):
    """INVERTIBLE-COUPLING denoiser (InvDN / flow family).

    The body is a chain of additive coupling steps -- the building block of normalizing
    flows. DenoiseHalfInstNorm also splits the channel axis, but only to normalise one half;
    nothing crosses. Here the halves ALTERNATELY TRANSFORM EACH OTHER, so the graph is a
    braid: chunk, cross-half conv, cat, repeated. No other family has cross-half conditioning,
    and the exactly-invertible body means the network cannot lose information before the
    final projection -- the flow-family argument for why these denoise well.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.k1 = CouplingBlock(f)
        self.k2 = CouplingBlock(f)
        self.k3 = CouplingBlock(f)
        self.k4 = CouplingBlock(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.k1(h)
        h = self.k2(h)
        h = self.k3(h)
        h = self.k4(h)
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
            loss = self.criterion_mse(preds, clean) * 1000 + \
                self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
