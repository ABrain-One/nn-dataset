import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class MemoryGate(nn.Module):
    """Fuses ALL previous recursion states, not just the last one (MemNet's gate unit).

    A 1x1 convolution over the concatenation is what makes the connectivity dense: state k
    can reach the output without passing through states k+1..n, so early features survive
    depth instead of being overwritten.
    """

    def __init__(self, channels, n_states):
        super().__init__()
        self.fuse = nn.Conv2d(channels * n_states, channels, 1)

    def forward(self, states):
        return self.fuse(torch.cat(states, dim=1))


class Net(nn.Module):
    """RECURSIVE denoiser with DENSE MEMORY (MemNet family).

    Two properties no model in the corpus has:

    WEIGHT REUSE. One recursive unit is applied N times. Every other family in the pool has
    independent weights per block, so depth costs parameters; here depth is free and the GA
    can mutate the recursion COUNT as a hyperparameter of the architecture -- a mutation axis
    that does not exist anywhere else in the seed pool.

    DENSE INTER-STATE CONNECTIVITY. Each memory gate sees every prior state, so the graph is
    not a chain. Under the WL hash this is a genuinely different topology from the corpus's
    encoder-decoder skips, which only ever connect matching resolutions.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 20
        self.n_rec = 4
        self.head = nn.Conv2d(3, f, 3, padding=1)
        # ONE recursive unit, applied n_rec times -- this is the weight reuse.
        self.rec = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1))
        self.act = nn.ReLU(inplace=True)
        self.gates = nn.ModuleList(
            [MemoryGate(f, i + 2) for i in range(self.n_rec)])
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        states = [h]
        for i in range(self.n_rec):
            r = self.act(self.rec(states[-1]) + states[-1])
            states.append(r)
            # The gate consumes the whole history, so every earlier state has a direct path
            # to this point.
            h = self.gates[i](states)
            states[-1] = h
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

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
            # Tighter than the corpus default of 0.1: a reused weight receives n_rec
            # gradient contributions per step, so the effective gradient is several times
            # larger than an equivalent non-recursive stack would produce.
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
