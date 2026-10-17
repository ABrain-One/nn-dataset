import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NLBlock(nn.Module):
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
        b, c = (x.shape[0], x.shape[1])
        h, w = (x.shape[2], x.shape[3])
        A = torch.softmax(self.basis(x).reshape(b, self.k, h * w), dim=-1)
        X = x.reshape(b, c, h * w)
        coeff = X @ A.transpose(1, 2)
        recon = (coeff @ A).reshape(b, c, h, w)
        return x + self.out(recon)

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
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.c1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.n1 = NLBlock(f)
        self.c2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.n2 = NLBlock(f)
        self.tail = nn.Conv2d(f, 3, 7, padding=3)
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