import torch
import torch.nn as nn
import torch.optim as optim

class HINBlock(nn.Module):
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
    """TWO-SUBNETWORK denoiser with HALF INSTANCE NORM and cross-stage feature fusion.

    Two structural properties, neither present in the corpus:

    THE SPLIT NORMALISATION described above -- the channel axis is partitioned and only one
    partition is normalised, so the graph contains a split/concat pair inside every block.

    CROSS-STAGE FUSION -- the second subnetwork receives the first subnetwork's FEATURES, not
    only its output image. DenoiseMultiStage passes a gated feature map between stages via an
    attention mask on an intermediate image; here the raw features are added directly, which
    is a different edge in the graph.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 24
        self.head1 = nn.Conv2d(3, f, 3, padding=1)
        self.s1a = HINBlock(f)
        self.s1b = HINBlock(f)
        self.out1 = nn.Conv2d(f, 3, 3, padding=1)
        self.head2 = nn.Conv2d(3, f, 3, padding=1)
        self.bridge = nn.Conv2d(f, f, 1)
        self.s2a = HINBlock(f)
        self.s2b = HINBlock(f)
        self.out2 = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self._mid = None
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head1(x)
        h = self.s1a(h)
        h = self.s1b(h)
        img1 = self.out1(h) + x
        self._mid = img1
        g = self.head2(torch.clamp(img1, 0.0, 1.0)) + self.bridge(h)
        g = self.s2a(g)
        g = self.s2b(g)
        return torch.clamp(self.out2(g) + x, 0.0, 1.0)

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
            if self._mid is not None:
                loss = loss + self.criterion_l1(torch.clamp(self._mid, 0.0, 1.0), clean) * 30
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()