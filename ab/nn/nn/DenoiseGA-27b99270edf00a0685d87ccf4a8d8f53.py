import torch
import torch.nn as nn
import torch.optim as optim

class AxialBlock(nn.Module):
    """Fast-Fourier-Convolution unit (FFC / LaMa / DeepRFT family): transform to the
    frequency domain, apply a learned pointwise mix to the spectrum, transform back. One
    frequency-domain multiply touches EVERY pixel, so the receptive field is global in a
    single op rather than grown layer by layer."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        h, w = (x.shape[2], x.shape[3])
        spec = torch.fft.rfft2(x, norm='ortho')
        z = torch.view_as_real(spec)
        b, c = (z.shape[0], z.shape[1])
        fh, fw = (z.shape[2], z.shape[3])
        z = z.permute(0, 1, 4, 2, 3).reshape(b, c * 2, fh, fw)
        z = self.act(self.mix(z))
        z = z.reshape(b, c, 2, fh, fw).permute(0, 1, 3, 4, 2).contiguous()
        out = torch.fft.irfft2(torch.view_as_complex(z), s=(h, w), norm='ortho')
        return out + x

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

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.stem = nn.Sequential(nn.Conv2d(3, f, 3, stride=2, padding=1), nn.SiLU(inplace=True), nn.Conv2d(f, f, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.ax1 = AxialBlock(f)
        self.ax2 = AxialBlock(f)
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(f, 24, 3, padding=1), nn.ReLU(inplace=True))
        self.local = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True))
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