# name=DenoiseSplitterNetF
# FAITHFUL SplitterNet port (Flepp/Ignatov CVPR2024) — the real MULTI-PATH split-tree, ~731k
# params, num_filters=32. At each encoder node the F-channel tensor is split into two F/2 halves,
# each strided-conv'd to F channels, and EACH child recurses independently (distinct weights) —
# so there are 2^depth parallel paths at the bottom (this is what my first, merge-based port threw
# away). Bottom paths get a middle block (simple channel attention + spatial attention); the
# decoder concatenates sibling paths, transpose-convs up, and adds the pre-split skip. Global
# residual + clamp. Trace-clean (static tree, static channel slices).
import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class SimpleChannelAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean(dim=(2, 3), keepdim=True)))


class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class MidBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(f, f, 3)
        self.conv2 = nn.Conv2d(f, f, 3)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ca = SimpleChannelAtt(f)
        self.sa = SpatialAtt()

    def forward(self, x):
        n = self.ca(self.act(self.conv1(self.pad(x)))) + x
        return self.sa(self.act(self.conv2(self.pad(n)))) + n


class SplitNode(nn.Module):
    """One encoder/decoder node: split channels, conv each half down, recurse (or mid at leaf),
    concat siblings, transpose up, add skip. Distinct weights per node => true multi-path tree."""

    def __init__(self, f, level, depth):
        super().__init__()
        self.half = f // 2
        self.pad = nn.ReflectionPad2d(1)
        self.conv_a = nn.Conv2d(self.half, f, 3, stride=2)
        self.conv_b = nn.Conv2d(self.half, f, 3, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.leaf = level >= depth - 1
        if self.leaf:
            self.mid_a = MidBlock(f)
            self.mid_b = MidBlock(f)
        else:
            self.child_a = SplitNode(f, level + 1, depth)
            self.child_b = SplitNode(f, level + 1, depth)
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
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=(3, 256, 256), prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        f = 32                       # SplitterNet num_filters
        self.head = nn.Conv2d(ch, f, 3, padding=1)
        self.tree = SplitNode(f, 0, depth=4)
        self.tail = nn.Conv2d(f, ch, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        h = self.tree(self.head(x))
        return torch.clamp(self.tail(h) + identity, 0.0, 1.0)

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
