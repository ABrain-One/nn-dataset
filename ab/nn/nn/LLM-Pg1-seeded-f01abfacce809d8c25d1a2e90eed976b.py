import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr"}


class HaarDown(nn.Module):
    def __init__(self, c):
        super().__init__()
        base = torch.tensor([[[1., 1.], [1., 1.]],
                             [[1., -1.], [1., -1.]],
                             [[1., 1.], [-1., -1.]],
                             [[1., -1.], [-1., 1.]]], dtype=torch.float32) / 2.0
        self.register_buffer("w", base.unsqueeze(1).repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        return F.conv2d(x, self.w, stride=2, groups=self.c)


class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion_factor=4):
        super().__init__()
        hidden_dim = channels * expansion_factor
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.scale = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))
        out = self.conv3(self.act(out))
        return self.scale(out + identity)


class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        self.ha1 = HaarDown(ch)
        self.ha2 = HaarDown(4 * ch)
        f = 16 * ch
        self.enc = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            ResidualBlock(f),
            ResidualBlock(f),
            ResidualBlock(f),
            AttentionBlock(f)
        )
        self.dec = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.shortcut = nn.Conv2d(ch, 16, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(16 + ch, 32, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(32, ch, kernel_size=3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        x = self.enc(self.ha2(self.ha1(x)))
        x = self.body(x)
        x = self.dec(x)
        x = self.up2(self.up1(x))
        shortcut = self.shortcut(identity)
        shortcut = self.residual(shortcut)
        x = self.relu(torch.cat([shortcut, x], dim=1))
        x = self.conv_out(x)
        x = self.conv_final(x)
        return torch.clamp(x + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)

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
                    m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
