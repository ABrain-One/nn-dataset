import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)



class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        n_channels = in_shape[1]

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_channels)

        self.to(self.device)

        self._init_optimizer(prm)
        self.criterion = nn.MSELoss()

    def _init_optimizer(self, prm):
        lr = prm.get("lr", 1e-3)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        inp = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        noise = self.outc(x)

        return torch.clamp(inp + noise, 0.0, 1.0)

    def train_setup(self, prm):
        self._init_optimizer(prm)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0

        for noisy, clean in train_data:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            pred = self(noisy)
            loss = self.criterion(pred, clean)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)
