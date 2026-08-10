import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class Net(nn.Module):
    """TWO INDEPENDENT FULL-DEPTH BRANCHES, fused only at the end (BRDNet family).

    The distinction from DenoiseParallelMR, the corpus's other parallel model, is that the
    branches here run at the SAME resolution and differ only in receptive field: one is a
    plain BN-CNN, the other is purely dilated. Neither ever sees the other's activations --
    there is no cross-branch skip, no intermediate fusion, nothing until the concatenation
    that feeds the output layer.

    That makes the graph two long disjoint chains joined at one node, which is a shape the
    WL hash has no near neighbour for: every other multi-path model in the corpus exchanges
    information mid-stream.

    It also gives the GA a mutation surface with a property none of the others have --
    branches can be mutated INDEPENDENTLY without any shape coupling, because they only ever
    meet at a concatenation whose width is the sum of two independent widths.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 20

        plain = [nn.Conv2d(3, f, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(5):
            plain += [nn.Conv2d(f, f, 3, padding=1), nn.BatchNorm2d(f),
                      nn.ReLU(inplace=True)]
        self.branch_plain = nn.Sequential(*plain)

        dil = [nn.Conv2d(3, f, 3, padding=1), nn.ReLU(inplace=True)]
        for d in (2, 3, 4, 3, 2):
            dil += [nn.Conv2d(f, f, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
        self.branch_dilated = nn.Sequential(*dil)

        # The ONLY point the two branches meet.
        self.fuse = nn.Conv2d(f * 2, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        a = self.branch_plain(x)
        b = self.branch_dilated(x)
        return torch.clamp(self.fuse(torch.cat([a, b], dim=1)) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5)

    def learn(self, train_data):
        self.train()
        bn_layers = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + \
                self.criterion_l1(preds, clean) * 50
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
