import torch
import torch.nn as nn
import torch.optim as optim

class Expert(nn.Module):
    """Fast-Fourier-Convolution unit (FFC / LaMa / DeepRFT family): transform to the
    frequency domain, apply a learned pointwise mix to the spectrum, transform back. One
    frequency-domain multiply touches EVERY pixel, so the receptive field is global in a
    single op rather than grown layer by layer."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, 1)
        self.act = nn.SiLU()

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
    """PER-PIXEL MIXTURE-OF-EXPERTS denoiser (Path-Restore / dynamic-routing family).

    Three full expert stacks run in parallel and a routing head -- computed from the NOISY
    INPUT, not from features -- blends them with per-pixel softmax weights. Flat regions can
    route to one expert while textured regions route to another, inside a single image.

    Distinct from DenoiseDualBranch: there two branches are CONCATENATED once and fused by a
    learned conv -- every pixel gets the same fixed mixing. Here the mixing weights are a
    softmax field that varies per pixel, so the graph carries an elementwise-multiply gate
    structure rather than a concat. Distinct from DenoiseKernelMix by granularity: that
    gates single kernels inside a block; this routes whole experts once.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 20
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.e1 = Expert(f)
        self.e2 = Expert(f)
        self.e3 = Expert(f)
        self.router = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(12, 3, 1))
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        w = torch.softmax(self.router(x), dim=1)
        mixed = self.e1(h) * w[:, 0:1] + self.e2(h) * w[:, 1:2] + self.e3(h) * w[:, 2:3]
        return torch.clamp(self.tail(mixed) + x, 0.0, 1.0)

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