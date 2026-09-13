import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class SAM(nn.Module):
    """Supervised Attention Module (MPRNet). The stage emits an IMAGE, that image is
    supervised, and an attention mask derived from it gates the features handed to the next
    stage. This is what makes the topology multi-stage rather than merely deep: the signal
    passed forward is filtered by an intermediate prediction, not just more features."""

    def __init__(self, channels):
        super().__init__()
        self.to_img = nn.Conv2d(channels, 3, 3, padding=1)
        self.from_img = nn.Conv2d(3, channels, 3, padding=1)
        self.gate = nn.Conv2d(channels, channels, 1)

    def forward(self, feat, inp):
        img = self.to_img(feat) + inp
        mask = torch.sigmoid(self.gate(self.from_img(img)))
        return feat * mask + feat, img


class Stage(nn.Module):
    def __init__(self, channels, depth=3):
        super().__init__()
        body = []
        for _ in range(depth):
            body += [nn.Conv2d(channels, channels, 3, padding=1),
                     nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x


class Net(nn.Module):
    """MULTI-STAGE PROGRESSIVE denoiser (MPRNet family).

    Distinct from everything in the corpus by MACRO-TOPOLOGY, not by block choice. The corpus
    holds encoder-decoder U-Nets, one flat dilated net, a Laplacian pyramid and a parallel
    multi-resolution net -- all of which restore the image ONCE. This restores it THREE times:
    each stage produces a full-resolution image, that image is supervised, and a mask derived
    from it gates the features entering the next stage.

    The intermediate images are returned only during training; forward() emits the final one,
    so the evaluation contract is unchanged.
    """

    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        f = 20
        self.head = nn.Conv2d(3, f, 3, padding=1)
        self.s1 = Stage(f)
        self.sam1 = SAM(f)
        self.s2 = Stage(f)
        self.sam2 = SAM(f)
        self.s3 = Stage(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self._aux = None
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.s1(h)
        h, img1 = self.sam1(h, x)
        h = self.s2(h)
        h, img2 = self.sam2(h, x)
        h = self.s3(h)
        out = torch.clamp(self.tail(h) + x, 0.0, 1.0)
        # Stashed rather than returned: the harness expects a single tensor from forward,
        # and learn() is the only consumer of the intermediate supervision.
        self._aux = (img1, img2)
        return out

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
            # DEEP SUPERVISION is the point of the architecture: without a loss on the
            # intermediate images the SAM masks get no gradient signal of their own and the
            # three stages collapse into one deep stack.
            if self._aux is not None:
                for img in self._aux:
                    loss = loss + self.criterion_l1(
                        torch.clamp(img, 0.0, 1.0), clean) * 25
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
