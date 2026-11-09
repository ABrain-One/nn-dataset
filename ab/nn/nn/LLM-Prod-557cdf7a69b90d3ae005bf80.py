import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'lr'}


class LiteCtxBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ctx_avg = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.ctx = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.PReLU(channels)

    def forward(self, x):
        out = self.act(self.conv1(x))
        ctx = self.ctx(self.ctx_avg(out))
        ctx = F.interpolate(ctx, size=out.shape[-2:], mode='nearest')
        return self.conv2(out + ctx) + x


class MultiScaleResidualNetwork(nn.Module):
    def __init__(self, num_filters=16):
        super().__init__()
        f = num_filters

        self.act = nn.ReLU(inplace=True)

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





class LiteCtxSplitNet(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        channels = 3
        f0 = 16
        f1, f2, f3, f4 = (f0 * 2, f0 * 4, f0 * 8, f0 * 16)
        self.in_conv = nn.Conv2d(channels, f0, 3, padding=1)
        self.eb0 = LiteCtxBlock(f0)
        self.down0 = nn.Sequential(nn.Conv2d(f0, f1, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb1 = LiteCtxBlock(f1)
        self.down1 = nn.Sequential(nn.Conv2d(f1, f2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb2 = LiteCtxBlock(f2)
        self.down2 = nn.Sequential(nn.Conv2d(f2, f3, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.eb3 = LiteCtxBlock(f3)
        self.down3 = nn.Sequential(nn.Conv2d(f3, f4, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = LiteCtxBlock(f4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Conv2d(f4 + f3, f3, 3, padding=1)
        self.db3 = LiteCtxBlock(f3)
        self.up2 = nn.Conv2d(f3 + f2, f2, 3, padding=1)
        self.db2 = LiteCtxBlock(f2)
        self.up1 = nn.Conv2d(f2 + f1, f1, 3, padding=1)
        self.db1 = LiteCtxBlock(f1)
        self.up0 = nn.Conv2d(f1 + f0, f0, 7, padding=3)
        self.db0 = LiteCtxBlock(f0)
        self.out_conv = nn.Conv2d(f0, channels, 7, padding=3)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(self.device)

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.in_conv(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        e3 = self.eb3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.db3(self.up3(torch.cat([self.upsample(b), e3], 1)))
        d2 = self.db2(self.up2(torch.cat([self.upsample(d3), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.upsample(d2), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.upsample(d1), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

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


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        self.net = MultiScaleResidualNetwork(num_filters=16)
        self.ctx_net = LiteCtxSplitNet(in_shape, None, prm, device)
        self.criterion = nn.MSELoss()

        self.train_setup(prm)
        self.to(self.device)

    def train_setup(self, prm):
        lr = prm.get("lr", 1e-4)
        batch_size = prm.get("batch", 16)
        epoch_max = prm.get("epoch_max", 50)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch_max, eta_min=1e-5)

    def forward(self, x):
        resnet_out = torch.clamp(self.net(x), 0.0, 1.0)
        ctx_out = torch.clamp(self.ctx_net(x), 0.0, 1.0)
        return 0.5 * (resnet_out + ctx_out)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        self.scheduler.step()
        return total_loss / max(count, 1)
