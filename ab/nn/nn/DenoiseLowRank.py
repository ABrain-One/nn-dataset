import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class LowRankBlock(nn.Module):
    """Spatial low-rank projection (NBNet family): a small head predicts K spatial basis
    maps, features are projected onto that K-dimensional subspace and reconstructed. The
    bottleneck forces a globally-consistent estimate -- noise, being spatially incoherent,
    does not survive a rank-8 spatial representation; structure does."""

    def __init__(self, channels):
        super().__init__()
        self.k = 8
        self.basis = nn.Conv2d(channels, self.k, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        h, w = x.shape[2], x.shape[3]
        # each basis map normalised into a spatial distribution
        A = torch.softmax(self.basis(x).reshape(b, self.k, h * w), dim=-1)   # (B,K,HW)
        X = x.reshape(b, c, h * w)                                           # (B,C,HW)
        coeff = X @ A.transpose(1, 2)                                        # (B,C,K)
        recon = (coeff @ A).reshape(b, c, h, w)                              # rank-K field
        return x + self.out(recon)


class Net(nn.Module):
    """LOW-RANK SUBSPACE denoiser (NBNet family).

    The corpus reduces noise by local filtering, multi-scale fusion, or attention. This
    family does it by RANK: inside every block the spatial feature field is projected onto
    just K=8 learned basis maps and reconstructed, so the graph contains batched matmuls
    over (C x HW)(HW x K) -- a global bilinear structure that neither convolution nor
    axial/windowed attention produces. The rank K is a mutation axis unique to this family.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.c1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.r1 = LowRankBlock(f)
        self.c2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.r2 = LowRankBlock(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.r1(self.c1(h))
        h = self.r2(self.c2(h))
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
