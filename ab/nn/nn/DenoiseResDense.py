import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class RDB(nn.Module):
    """Residual Dense Block (RDN).

    Inside the block, every convolution sees the concatenation of ALL previous outputs, so
    the layer widths grow arithmetically and a 1x1 fuses them back down. This is dense
    connectivity WITHIN a block, which is a different graph from DenoiseRecMem's dense
    connectivity ACROSS recursions: there the states are the same width and gated, here the
    width grows and is fused.
    """

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


class Net(nn.Module):
    """RESIDUAL DENSE NETWORK with GLOBAL FEATURE FUSION (RDN family).

    Two levels of aggregation, and the outer one is what the corpus lacks: every block's
    output is retained and ALL of them are concatenated and fused at the end, so the network
    has a global path from each block directly to the output. Combined with the intra-block
    dense connectivity, the graph has a density of skip edges that nothing else in the pool
    approaches -- the U-Nets connect matching resolutions only, and the flat models are
    chains.

    Growth rate is a mutation axis unique to this family: a descendant can change how fast
    the intra-block width grows without altering the block count or the base width.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f, g, n_blocks = 20, 12, 3
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.blocks = nn.ModuleList([RDB(f, g) for _ in range(n_blocks)])
        # GLOBAL fusion over every block output.
        self.gff = nn.Sequential(
            nn.Conv2d(f * n_blocks, f, 1),
            nn.Conv2d(f, f, 3, padding=1))
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        outs = []
        cur = h
        for b in self.blocks:
            cur = b(cur)
            outs.append(cur)
        fused = self.gff(torch.cat(outs, dim=1)) + h
        return torch.clamp(self.tail(fused) + x, 0.0, 1.0)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
