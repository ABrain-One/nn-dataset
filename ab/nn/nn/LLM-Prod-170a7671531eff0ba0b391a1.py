import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class GroupedConv(nn.Module):
    """Depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=None):
        super().__init__()
        groups = groups or in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FourierUnit(nn.Module):
    """Fourier unit for frequency-domain denoising."""

    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.t1 = nn.Sequential(nn.Conv2d(half, half, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(half, half, 3, padding=1))
        self.t2 = nn.Sequential(nn.Conv2d(half, half, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(half, half, 3, padding=1))

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        b = b + self.t1(a)
        a = a + self.t2(b)
        return x + torch.cat([a, b], dim=1)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 16 * in_shape[1]
        self.head = GroupedConv(in_shape[1], f, 7, padding=3)
        self.s1 = nn.Sequential(GroupedConv(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.f1 = FourierUnit(f)
        self.s2 = nn.Sequential(GroupedConv(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.f2 = FourierUnit(f)
        self.tail = GroupedConv(f, in_shape[1], 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.f1(self.s1(h))
        h = self.f2(self.s2(h))
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        bn_layers = [m for m in self.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all((torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all() for m in bn_layers))
            if bad:
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
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
