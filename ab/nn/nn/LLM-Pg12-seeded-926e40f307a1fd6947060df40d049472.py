import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def supported_hyperparameters():
    return {"lr"}

class DualBlock(nn.Module):
    """Dual path block combining residual and SE blocks."""

    def __init__(self, channels, se_ratio=16):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // se_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_out = self.residual(x)
        se_out = self.se(res_out)
        return res_out * se_out

class MultiPathTree(nn.Module):
    """Multi-path tree structure inspired by SplitterNet."""

    def __init__(self, f, level, depth):
        super().__init__()
        self.half = f // 2
        self.pad = nn.ReflectionPad2d(1)
        self.conv_a = nn.Conv2d(self.half, f, 3, stride=2)
        self.conv_b = nn.Conv2d(self.half, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.leaf = level >= depth - 1
        if self.leaf:
            self.mid_a = DualBlock(f)
            self.mid_b = DualBlock(f)
        else:
            self.child_a = MultiPathTree(f, level + 1, depth)
            self.child_b = MultiPathTree(f, level + 1, depth)
        self.up = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        skip = x
        a = self.act(self.conv_a(self.pad(x[:, :self.half])))
        b = self.act(self.conv_b(self.pad(x[:, self.half:])))
        if self.leaf:
            a = self.mid_a(a); b = self.mid_b(b)
        else:
            a = self.child_a(a); b = self.child_b(b)
        m = self.act2(self.up(torch.cat([a, b], dim=1)))
        return m + skip

class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        f = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.in_conv = nn.Conv2d(3 * 4, f, 3, padding=1)
        self.mid_block = DualBlock(f)
        self.tree = MultiPathTree(f, 0, depth=4)
        self.tail = nn.Conv2d(f, 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.in_conv(self.unshuffle(x))
        h = self.mid_block(h)
        h = self.tree(h)
        return torch.clamp(self.shuffle(self.tail(h)) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        batch_size = prm.get("batch", 16)
        epoch_max = prm.get("epoch_max", 50)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch_max, eta_min=1e-5)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
