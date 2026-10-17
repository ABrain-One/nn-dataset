"""Pure NAFNet denoiser (Nonlinear Activation-Free Network).

A faithful, mobile-scaled NAFNet (Chen et al., ECCV 2022) as a *distinct* macro-family
seed for the denoise NAS pool. The existing pool is vanilla/ResNet U-Nets; the only
NAFNet-flavoured member (DenoiseUnet9F) uses an ad-hoc gated-adaptive block. This uses
the canonical NAFBlock — channels-first LayerNorm, SimpleGate (no ReLU/GELU anywhere),
Simplified Channel Attention, depthwise conv, and layer-scaled residuals — with
strided-conv downsampling and PixelShuffle upsampling.

Edge-legal: width/depth scaled so MACs stay well under the 9053-MMac cap (DenoiseUnet9F's
cost). Net interface matches the LEMUR denoise contract:
    Net(in_shape=(B,C,H,W), out_shape, prm, device); forward(noisy)->clean in [0,1].
"""
import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm over the channel dim (NAFNet's normalisation)."""

    def __init__(self, channels, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class SimpleGate(nn.Module):
    """NAFNet's activation-free gate: split channels in half and multiply."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b

class NAFBlock(nn.Module):
    """Canonical NAFNet block. No ReLU/GELU — nonlinearity comes from SimpleGate + SCA."""

    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_c = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_c // 2, dw_c // 2, 1))
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1)
        ffn_c = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1)
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        return x + y * self.gamma

class Net(nn.Module):

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        ch = in_shape[1] if len(in_shape) >= 3 else 3
        width = 32
        enc_blks = [1, 1, 1]
        mid_blks = 1
        dec_blks = [1, 1, 1]
        self.intro = nn.Conv2d(ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, ch, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        c = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c = c * 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(mid_blks)])
        for n in dec_blks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c = c // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.padder = 2 ** len(enc_blks)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def _pad(self, x):
        _, _, h, w = x.shape
        p = self.padder
        ph = (p - h % p) % p
        pw = (p - w % p) % p
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph))
        return (x, h, w)

    def forward(self, x):
        identity = x
        y, h, w = self._pad(x)
        y = self.intro(y)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)
        y = self.middle(y)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            y = up(y)
            y = y + skip
            y = dec(y)
        y = self.ending(y)
        y = y[:, :, :h, :w]
        return torch.clamp(y + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
