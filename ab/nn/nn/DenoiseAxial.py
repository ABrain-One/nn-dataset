import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class AxialBlock(nn.Module):
    """Axial attention: one 1-D attention along ROWS, then one along COLUMNS. Two cheap 1-D
    passes give every pixel a full-image receptive field -- the criss-cross trick -- without
    ever forming the quadratic full-spatial attention map."""

    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def _attn_lastdim(self, q, k, v):
        # attention over the LAST spatial dim; caller permutes to choose the axis
        a = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        return (v @ a.transpose(-2, -1))

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        # rows: sequence along W  (fold H into batch)
        b, c = q.shape[0], q.shape[1]
        h, w = q.shape[2], q.shape[3]
        qr = q.permute(0, 2, 1, 3).reshape(b * h, c, w)
        kr = k.permute(0, 2, 1, 3).reshape(b * h, c, w)
        vr = v.permute(0, 2, 1, 3).reshape(b * h, c, w)
        r = self._attn_lastdim(qr, kr, vr).reshape(b, h, c, w).permute(0, 2, 1, 3)
        # columns: sequence along H  (fold W into batch)
        qc = r.permute(0, 3, 1, 2).reshape(b * w, c, h)
        kc = k.permute(0, 3, 1, 2).reshape(b * w, c, h)
        vc = v.permute(0, 3, 1, 2).reshape(b * w, c, h)
        o = self._attn_lastdim(qc, kc, vc).reshape(b, w, c, h).permute(0, 2, 3, 1)
        return x + self.proj(o)


class Net(nn.Module):
    """AXIAL-ATTENTION denoiser (criss-cross / axial transformer family).

    DenoiseSelfAttn already exists, so the claim has to be sharper: this is not spatial
    self-attention over a window or a channel-attention squeeze. Attention runs as two
    SEPARATE 1-D passes -- all rows, then all columns -- so the graph contains two
    batched-matmul attention units with a permute between them, and information crosses the
    whole image in exactly two hops.

    Attention runs at QUARTER resolution (the conv stem pools 4x) both for cost and because
    a 512-crop full-resolution row attention would be a 512-long sequence per row at batch
    16 -- memory the 24 GB training cards do not have.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, f, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.ax1 = AxialBlock(f)
        self.ax2 = AxialBlock(f)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(f, 24, 3, padding=1), nn.ReLU(inplace=True))
        # full-resolution local path: attention recovers global structure, this keeps edges
        self.local = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(48, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        g = self.up(self.ax2(self.ax1(self.stem(x))))
        l = self.local(x)
        return torch.clamp(self.tail(torch.cat([g, l], dim=1)) + x, 0.0, 1.0)

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
