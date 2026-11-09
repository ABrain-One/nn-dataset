import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr'}

class LiteDenoisingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channels, 5, padding=2)
        self.actv = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        identity = x
        out = self.actv(self.conv1(x))
        out = self.conv2(out)
        return self.actv(out + identity)

class ELUAvgBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        out = self.act(self.conv1(x))
        c = self.ctx(self.ctx_pool(out))
        c = nn.functional.interpolate(c, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + c) + x

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3 = f0 * 2, f0 * 4, f0 * 8
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = LiteDenoisingBlock(f0, f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = LiteDenoisingBlock(f1, f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = LiteDenoisingBlock(f2, f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = LiteDenoisingBlock(f3, f3)
        self.elu_avg_bottleneck = ELUAvgBlock(f3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.elu_avg_up2 = ELUAvgBlock(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.elu_avg_up1 = ELUAvgBlock(f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 3, padding=1)
        self.elu_avg_up0 = ELUAvgBlock(f0)
        self.out_conv = nn.Conv2d(f0, channels, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        b = self.elu_avg_bottleneck(self.down2(e2))
        d2 = self.elu_avg_up2(self.up2(torch.cat([self.upsample(b), e2], 1)))
        d1 = self.elu_avg_up1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.elu_avg_up0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
