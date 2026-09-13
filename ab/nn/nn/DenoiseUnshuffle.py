import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class Net(nn.Module):
    """PIXEL-UNSHUFFLE denoiser (FFDNet family).

    The corpus reaches low resolution by POOLING or STRIDING, which discards information, and
    returns via interpolation or transposed convolution, which invents it. This does neither.
    A pixel-unshuffle folds a 2x2 neighbourhood into the CHANNEL axis, so the tensor is a
    quarter of the area with four times the channels and NOTHING IS LOST -- the whole body
    runs at half resolution, and pixel-shuffle folds it back exactly.

    That makes it cheap in the way the MAC cap rewards (the body sees a quarter of the
    pixels) while remaining information-preserving, and it is a distinct macro-topology: no
    encoder-decoder, no cross-scale skips, no interpolation anywhere in the graph.

    DenoiseSubPixel already uses pixel-shuffle, but as an UPSAMPLER inside a U-Net decoder.
    Unshuffling FIRST, before any convolution, is the structural difference -- the entire
    network lives in the folded space.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 40
        self.down = nn.PixelUnshuffle(2)          # 3 -> 12 channels, HxW -> H/2 x W/2
        self.up = nn.PixelShuffle(2)              # exact inverse
        self.head = nn.Conv2d(12, f, 3, padding=1)
        body = []
        for _ in range(6):
            body += [nn.Conv2d(f, f, 3, padding=1),
                     nn.BatchNorm2d(f),
                     nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        # 12 output channels so pixel-shuffle returns exactly 3.
        self.tail = nn.Conv2d(f, 12, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        # NO DATA-DEPENDENT BRANCH. An `if` on a traced value makes the whole model
        # untraceable by torch.fx -- the diversity gate then has no graded opinion on it OR
        # on any of its descendants, which silently weakens the gate for an entire lineage.
        # One such model already killed a GA run outright with
        # "symbolically traced variables cannot be used as inputs to control flow".
        # Every crop this pipeline uses is even (256 and 512), so the fold is always exact
        # and the padding branch bought nothing.
        d = self.down(x)
        y = self.head(d)
        y = self.body(y)
        y = self.up(self.tail(y)) + x
        return torch.clamp(y, 0.0, 1.0)

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
            # BN buffers are updated by the forward BEFORE the loss is checked, so a batch
            # that produces a non-finite loss has already poisoned eval unless the buffers
            # are restored. Same guard the corpus's BN-bearing seeds use.
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
