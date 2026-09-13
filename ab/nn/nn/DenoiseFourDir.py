import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class DirStack(nn.Module):
    """The ONE shared directional stack. Its receptive field is biased strictly toward one
    half-plane by asymmetric padding: pad two rows above, crop two below, so every conv
    reaches only upward context. The caller rotates the input so this same stack scans each
    of the four directions."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=(0, 1))
        self.c2 = nn.Conv2d(channels, channels, 3, padding=(0, 1))
        self.act = nn.ReLU(inplace=True)
        self.pad = nn.ZeroPad2d((0, 0, 2, 0))          # top-only padding: upward context

    def forward(self, x):
        h = self.act(self.c1(self.pad(x)))
        h = self.act(self.c2(self.pad(h)))
        return h


class Net(nn.Module):
    """FOUR-DIRECTION PROPAGATION denoiser (IRNN / Spatial Propagation family).

    Context is gathered by FOUR DIRECTIONAL SWEEPS: the same weight-shared stack scans up,
    down, left and right (implemented as rot90 -> shared stack -> rot90 back), and a 1x1
    merges the four half-plane views. Every other family in the pool has an isotropic
    receptive field; here direction is an explicit axis of the graph -- four parallel paths
    that differ only by rotation, sharing one set of weights.

    That structure is what the propagation literature uses to carry information across long
    ranges at conv cost, and rotation-tied weights are a mutation surface nothing else has:
    untying them, or dropping to two directions, are one-line mutations with real meaning.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.dir = DirStack(f)                          # ONE stack, four uses
        self.merge = nn.Conv2d(f * 4, f, 1)
        self.mix = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        views = []
        for k in range(4):
            r = torch.rot90(h, k, dims=(2, 3))
            r = self.dir(r)
            views.append(torch.rot90(r, -k, dims=(2, 3)))
        m = self.mix(self.merge(torch.cat(views, dim=1)))
        return torch.clamp(self.tail(m) + x, 0.0, 1.0)

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
            # the shared stack receives four gradient contributions per step
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
