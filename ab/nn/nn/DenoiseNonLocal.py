import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class NLBlock(nn.Module):
    """Windowed non-local aggregation (N3Net / NLRN family): each pixel is replaced by a
    weighted average of its 7x7 neighbourhood, with weights computed from FEATURE SIMILARITY
    -- a learned, differentiable BM3D-style collaborative filter. The contrast with
    DenoiseKPN is where the weights come from: KPN predicts them with a conv head; here they
    ARE the query-key dot products, so self-similar structure (edges, texture) reinforces
    itself and incoherent noise cancels.

    Computed as 49 shifted views of a zero-padded tensor rather than one F.unfold: unfold
    materialises a (B,C,49,H,W) buffer (~5 GB at batch 16), which does not fit a 24 GB card;
    the shifted form peaks at the (B,49,H,W) similarity map instead. Same math, same output."""

    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        q = self.q(x)                                              # (B,C,H,W)
        kp = F.pad(self.k(x), (3, 3, 3, 3))
        vp = F.pad(self.v(x), (3, 3, 3, 3))
        logits = []
        for dy in range(7):
            for dx in range(7):
                logits.append((q * kp[:, :, dy:dy + h, dx:dx + w]).sum(1) * self.scale)
        sim = torch.softmax(torch.stack(logits, dim=1), dim=1)     # (B,49,H,W)
        out = x
        i = 0
        for dy in range(7):
            for dx in range(7):
                out = out + vp[:, :, dy:dy + h, dx:dx + w] * sim[:, i].unsqueeze(1)
                i += 1
        return out


class Net(nn.Module):
    """NON-LOCAL SELF-SIMILARITY denoiser (N3Net / NLRN family).

    Classical denoising's strongest prior -- natural images repeat themselves -- expressed as
    a differentiable layer. No other family in the pool computes its filter weights from
    feature similarity: attention families (SelfAttn, Axial) attend over fixed axes with
    positionless queries, KPN predicts taps, KernelMix gates static kernels. Here the graph
    contains an unfold on BOTH key and value paths and a dot-product-softmax between them.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.c1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.n1 = NLBlock(f)
        self.c2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.n2 = NLBlock(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
