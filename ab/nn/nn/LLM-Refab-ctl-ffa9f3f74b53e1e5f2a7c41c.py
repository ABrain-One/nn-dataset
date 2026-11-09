import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {"lr"}

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Conv2d(channels + i * growth, growth, 3, padding=1))
        self.act = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(channels + n_layers * growth, channels, 1)

    def forward(self, x):
        feats = [x]
        for conv in self.layers:
            feats.append(self.act(conv(torch.cat(feats, dim=1))))
        return self.fuse(torch.cat(feats, dim=1)) + x

class DilatedConvolutionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.d1(x))
        h = self.act(self.d2(h))
        return self.act(self.d4(h) + x)

class AdaptiveAverageBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 1, padding=0)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x

class AXIALAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def _attn_lastdim(self, q, k, v):
        a = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        return v @ a.transpose(-2, -1)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        b, c = q.shape[0], q.shape[1]
        h, w = q.shape[2], q.shape[3]
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
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, f, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(f, growth=f//4),
            ResidualDenseBlock(f, growth=f//4),
            ResidualDenseBlock(f, growth=f//4)
        ])
        
        self.dcbs = nn.ModuleList([
            DilatedConvolutionBlock(f),
            DilatedConvolutionBlock(f),
            DilatedConvolutionBlock(f)
        ])
        
        self.axbs = nn.ModuleList([
            AXIALAttentionBlock(f),
            AXIALAttentionBlock(f),
            AXIALAttentionBlock(f)
        ])
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(f, 24, 3, padding=1), nn.ReLU(inplace=True))
        
        self.local = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True))
        
        self.tail = nn.Conv2d(48, 3, 3, padding=1)
        
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        g = self.up(self.axbs[2](self.dcbs[2](self.rdbs[2](self.stem(x)))))
        l = self.local(x)
        return torch.clamp(self.tail(torch.cat([g, l], dim=1)) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
