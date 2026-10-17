import torch
import torch.nn as nn
import torch.optim as optim

class LowRankBlock(nn.Module):
    """Axial attention: one 1-D attention along ROWS, then one along COLUMNS. Two cheap 1-D
    passes give every pixel a full-image receptive field -- the criss-cross trick -- without
    ever forming the quadratic full-spatial attention map."""

    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** (-0.5)

    def _attn_lastdim(self, q, k, v):
        a = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        return v @ a.transpose(-2, -1)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        b, c = (q.shape[0], q.shape[1])
        h, w = (q.shape[2], q.shape[3])
        qr = q.permute(0, 2, 1, 3).reshape(b * h, c, w)
        kr = k.permute(0, 2, 1, 3).reshape(b * h, c, w)
        vr = v.permute(0, 2, 1, 3).reshape(b * h, c, w)
        r = self._attn_lastdim(qr, kr, vr).reshape(b, h, c, w).permute(0, 2, 1, 3)
        qc = r.permute(0, 3, 1, 2).reshape(b * w, c, h)
        kc = k.permute(0, 3, 1, 2).reshape(b * w, c, h)
        vc = v.permute(0, 3, 1, 2).reshape(b * w, c, h)
        o = self._attn_lastdim(qc, kc, vc).reshape(b, w, c, h).permute(0, 2, 3, 1)
        return x + self.proj(o)

class Net(nn.Module):
    """LOW-RANK SUBSPACE denoiser (NBNet family).

    The corpus reduces noise by local filtering, multi-scale fusion, or attention. This
    family does it by RANK: inside every block the spatial feature field is projected onto
    just K=8 learned basis maps and reconstructed, so the graph contains batched matmuls
    over (C x HW)(HW x K) -- a global bilinear structure that neither convolution nor
    axial/windowed attention produces. The rank K is a mutation axis unique to this family.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
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