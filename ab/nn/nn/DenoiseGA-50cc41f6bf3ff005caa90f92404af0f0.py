import torch
import torch.nn as nn
import torch.optim as optim

def supported_hyperparameters():
    return {'lr'}

class Prox(nn.Module):
    """The learned proximal operator. ONE instance, applied at every iteration."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(inplace=True, negative_slope=0.01), nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)

class Net(nn.Module):
    """DEEP UNFOLDING denoiser (DPIR / ISTA-Net family).

    The graph is an OPTIMISATION LOOP, not a feature extractor. Each iteration performs an
    explicit data-fidelity gradient step toward the observation, then applies a learned
    proximal operator -- the classic half-quadratic-splitting alternation, unrolled a fixed
    number of times with the step sizes as learned parameters.

    Nothing in the corpus has this shape. Every other family maps features forward once (or,
    in DenoiseMultiStage, three times); here the NOISY INPUT re-enters the computation at
    every iteration through the fidelity term, so the input is a source with as many
    consumers as there are iterations. That, plus a single reused proximal operator, gives a
    topology the WL hash sees as unlike anything in the pool.

    The learned step sizes are also a mutation surface the GA has never had: a descendant can
    change how far each iteration moves toward the data without touching a single layer.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.n_iter = 4
        self.prox = Prox(28)
        self.step = nn.Parameter(torch.full((self.n_iter,), 0.1))
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, y):
        x = y
        for k in range(self.n_iter):
            x = x - self.step[k] * (x - y)
            x = self.prox(x)
        return torch.clamp(x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.criterion_mse(preds, clean) * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
            self.optimizer.step()
            with torch.no_grad():
                self.step.clamp_(0.0, 1.0)