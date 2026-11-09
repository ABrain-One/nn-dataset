import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class KBNetDilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.basis = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])
        self.gate = nn.Conv2d(channels, 4, 1, padding=0)
        self.dil1 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.dil2 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=1)
        out = 0.0
        for i, conv in enumerate(self.basis):
            out = out + conv(x) * g[:, i:i + 1]
        dilated_out = self.dil1(x) + self.dil2(x)
        return x + self.act(out + dilated_out)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        f = 20
        self.head = nn.Conv2d(3, f, 1, padding=0)
        self.s1 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.kb1 = KBNetDilatedBlock(f)
        self.s2 = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True))
        self.kb2 = KBNetDilatedBlock(f)
        self.tail = nn.Conv2d(f, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.head(x)
        h = self.kb1(self.s1(h))
        h = self.kb2(self.s2(h))
        return torch.clamp(self.tail(h) + x, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
