import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class WeightedL1PSNRLoss(nn.Module):
    """
    L1 loss combined with an approximate PSNR-based weighting term.
    High-error regions contribute more to the gradient via the adaptive map.
    """
    def __init__(self, psnr_weight=0.3):
        super().__init__()
        self.psnr_weight = psnr_weight

    def forward(self, pred, target):
        diff = (pred - target).abs()
        l1 = diff.mean()
        err_map = diff.detach().mean(dim=1, keepdim=True)
        weight = 1.0 + self.psnr_weight * err_map / (err_map.mean() + 1e-6)
        return (diff * weight).mean()


class TriLevelUNet(nn.Module):
    def __init__(self, num_filters=16):
        super().__init__()
        f = num_filters

        self.act = nn.PReLU()

        self.conv1a = nn.Conv2d(3, f, 3, padding=1)
        self.conv1b = nn.Conv2d(f, f, 3, padding=1)
        self.conv1c = nn.Conv2d(f, f, 3, padding=1)

        self.conv2d = nn.Conv2d(f, f, 3, stride=2, padding=1)
        self.conv2a = nn.Conv2d(f, f, 3, padding=1)
        self.conv2b = nn.Conv2d(f, f, 3, padding=1)

        self.conv3d = nn.Conv2d(f, f, 3, stride=2, padding=1)
        self.conv3a = nn.Conv2d(f, f * 2, 3, padding=1)
        self.conv3b = nn.Conv2d(f * 2, f, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(f, f, 1, stride=2, output_padding=1)

        self.conv4a = nn.Conv2d(f * 2, f, 3, padding=1)
        self.conv4b = nn.Conv2d(f, f, 3, padding=1)
        self.conv4c = nn.Conv2d(f, f, 3, padding=1)

        self.up4 = nn.ConvTranspose2d(f, f, 1, stride=2, output_padding=1)

        self.conv5a = nn.Conv2d(f * 2, f, 3, padding=1)
        self.conv5b = nn.Conv2d(f, f, 3, padding=1)
        self.conv5c = nn.Conv2d(f, f, 3, padding=1)

        self.proj = nn.Conv2d(f, f, 1)
        self.conv6 = nn.Conv2d(f, f, 3, padding=1)
        self.final_conv = nn.Conv2d(f, 3, 3, padding=1)

    def forward(self, inp):
        a = self.act

        x1 = a(self.conv1a(inp))
        x = a(self.conv1c(a(self.conv1b(x1))))
        add1 = x1 + x

        x2 = self.conv2d(add1)
        x = a(self.conv2b(a(self.conv2a(x2))))
        add2 = x2 + x

        x3 = self.conv3d(add2)
        x = a(self.conv3b(a(self.conv3a(x3))))
        add3 = x3 + x

        x = self.up3(add3)
        x = self.conv4a(torch.cat([add2, x], dim=1))
        x2_ = a(x)
        x = a(self.conv4c(a(self.conv4b(x2_))))
        add4 = x2_ + x

        x = self.up4(add4)
        x = self.conv5a(torch.cat([add1, x], dim=1))
        x5 = a(x)
        x = a(self.conv5c(a(self.conv5b(x5))))
        add5 = x5 + x

        x = a(self.conv6(self.proj(add5)))
        return inp + self.final_conv(x)


class Net(nn.Module):
    """
    Compact three-level residual U-Net with PReLU activations, strided
    downsampling and transposed-conv upsampling, residual skip additions at
    each level and a global residual connection. Trained with an error-
    weighted L1 (PSNR-aware) loss.
    """
    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device="cuda"):
        super().__init__()
        self.device = device
        self.net = TriLevelUNet(num_filters=16)
        self.criterion = WeightedL1PSNRLoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()

            preds = self(noisy)
            loss = self.criterion(preds, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
