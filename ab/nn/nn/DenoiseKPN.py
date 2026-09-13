import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class KPNHead(nn.Module):
    """Predicts a per-pixel 3x3 filter (9 softmax weights) from features. The
    channel-preserving (channels) signature keeps it crossover-swappable."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_k = nn.Conv2d(channels, 9, 1)

    def forward(self, x):
        return torch.softmax(self.to_k(torch.relu(self.mix(x))), dim=1)


class Net(nn.Module):
    """KERNEL-PREDICTION denoiser (KPN family, burst-denoising lineage).

    The output is NOT produced by convolving learned static kernels with features. The
    network predicts a DIFFERENT 3x3 filter at every pixel, and the final image is the noisy
    input dynamically filtered by those predicted kernels. Every other model in the corpus
    applies the same weights everywhere; here the weights themselves are a function of the
    input, and the image-forming op is an unfold-multiply-sum, not a convolution.

    That gives the graph a shape nothing in the pool has -- the input feeds both the feature
    stack AND the final filtering op directly -- and gives the GA a new mutation surface:
    the predicted-kernel size is a knob that exists nowhere else.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 24
        self.feats = nn.Sequential(
            nn.Conv2d(3, f, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.head = KPNHead(f)
        self.refine = nn.Conv2d(3, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        h, w = x.shape[2], x.shape[3]
        k = self.head(self.feats(x))                      # (B, 9, H, W)
        patches = F.unfold(x, 3, padding=1)               # (B, C*9, H*W)
        patches = patches.view(b, c, 9, h, w)
        filtered = (patches * k.unsqueeze(1)).sum(2)      # per-pixel dynamic filter
        return torch.clamp(filtered + self.refine(filtered), 0.0, 1.0)

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
