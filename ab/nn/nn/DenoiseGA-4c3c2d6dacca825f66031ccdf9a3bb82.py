import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NLBlock(nn.Module):
    """Kernel-basis block (KBNet family): FOUR parallel convolutions whose outputs are
    mixed per-pixel by a learned softmax gate. The kernels are the basis; the gate picks a
    different combination at every location, so the effective filter is spatially adaptive
    while every weight stays static and countable."""

    def __init__(self, channels):
        super().__init__()
        self.basis = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])
        self.gate = nn.Conv2d(channels, 4, 5, padding=2)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=1)
        out = 0.0
        for i, conv in enumerate(self.basis):
            out = out + conv(x) * g[:, i:i + 1]
        return x + self.act(out + x)

class Net(nn.Module):
    """NON-LOCAL SELF-SIMILARITY denoiser (N3Net / NLRN family).

    Classical denoising's strongest prior -- natural images repeat themselves -- expressed as
    a differentiable layer. No other family in the pool computes its filter weights from
    feature similarity: attention families (SelfAttn, Axial) attend over fixed axes with
    positionless queries, KPN predicts taps, KernelMix gates static kernels. Here the graph
    contains an unfold on BOTH key and value paths and a dot-product-softmax between them.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 24
        self.head = nn.Conv2d(3, f, 1, padding=0)
        self.c1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.n1 = NLBlock(f)
        self.c2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.n2 = NLBlock(f)
        self.tail = nn.Conv2d(f, 3, 5, padding=2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.n1(self.c1(h))
        h = self.n2(self.c2(h))
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()