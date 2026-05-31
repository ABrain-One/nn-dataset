import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {"lr"}


@torch.no_grad()
def init_weights(m):
    """He (Kaiming) initialization for Conv2d, standard init for BatchNorm."""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.zeros_(m.bias)


class DownsampleBlock(nn.Module):
    """2x2 strided convolution with PReLU activation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    """Transposed convolution upsampling + skip concatenation + 3x3 fusion."""
    def __init__(self, in_channels, cat_channels, out_channels):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.actv_t = nn.PReLU(in_channels)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    """Initial RGB-to-feature mapping with two 3x3 convolutions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    """Final feature-to-RGB mapping with two 3x3 convolutions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    """
    Dense denoising block: four 3x3 convolutions with dense internal
    feature aggregation and a local residual connection. Intermediate
    features are of width `inner_channels` (typically in_channels // 2).
    """
    def __init__(self, in_channels, inner_channels, out_channels):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)
        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))
        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))
        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))
        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))
        return out_3 + x


class Net(nn.Module):
    """
    High-capacity U-Net-style denoising network (teacher model).

    Three-level encoder-bottleneck-decoder topology with feature widths
    of 64, 128, 256 in the encoder and 512 in the bottleneck. Uses
    transposed convolutions in the decoder for upsampling. Achieves
    37.71 dB PSNR on the Mobile AI 2026 validation benchmark but is
    not suitable for mobile NPU deployment due to its memory footprint;
    serves as the supervisor during knowledge distillation of the
    lightweight student (LiteDenoiseNet).

    Reference: "Real Image Denoising with Knowledge Distillation for
    High-Performance Mobile NPUs", CVPRW 2026.
    """
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        channels = in_shape[1]

        f0 = 64
        f1, f2, f3 = f0 * 2, f0 * 4, f0 * 8

        # Encoder
        self.input_block = InputBlock(channels, f0)
        self.eb0 = nn.Sequential(
            DenoisingBlock(f0, f0 // 2, f0), DenoisingBlock(f0, f0 // 2, f0)
        )
        self.down0 = DownsampleBlock(f0, f1)
        self.eb1 = nn.Sequential(
            DenoisingBlock(f1, f1 // 2, f1), DenoisingBlock(f1, f1 // 2, f1)
        )
        self.down1 = DownsampleBlock(f1, f2)
        self.eb2 = nn.Sequential(
            DenoisingBlock(f2, f2 // 2, f2), DenoisingBlock(f2, f2 // 2, f2)
        )
        self.down2 = DownsampleBlock(f2, f3)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DenoisingBlock(f3, f3 // 2, f3), DenoisingBlock(f3, f3 // 2, f3)
        )

        # Decoder
        self.up2 = UpsampleBlock(f3, f2, f2)
        self.db2 = nn.Sequential(
            DenoisingBlock(f2, f2 // 2, f2), DenoisingBlock(f2, f2 // 2, f2)
        )
        self.up1 = UpsampleBlock(f2, f1, f1)
        self.db1 = nn.Sequential(
            DenoisingBlock(f1, f1 // 2, f1), DenoisingBlock(f1, f1 // 2, f1)
        )
        self.up0 = UpsampleBlock(f1, f0, f0)
        self.db0 = nn.Sequential(
            DenoisingBlock(f0, f0 // 2, f0), DenoisingBlock(f0, f0 // 2, f0)
        )

        self.output_block = OutputBlock(f0, channels)

        self.apply(init_weights)

        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self._init_optimizer(prm)
        self.to(self.device)

    def _init_optimizer(self, prm):
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

    def train_setup(self, prm):
        self._init_optimizer(prm)

    def forward(self, x):
        identity = x
        e0 = self.eb0(self.input_block(x))
        e1 = self.eb1(self.down0(e0))
        e2 = self.eb2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.db2(self.up2([b, e2]))
        d1 = self.db1(self.up1([d2, e1]))
        d0 = self.db0(self.up0([d1, e0]))

        return torch.clamp(self.output_block(d0) + identity, 0.0, 1.0)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()

            pred = self(noisy)
            mse_loss = self.criterion_mse(pred, clean)
            l1_loss = self.criterion_l1(pred, clean)
            combined_loss = (mse_loss * 1000) + (l1_loss * 100)

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += mse_loss.item()
            count += 1

        return total_loss / max(count, 1)
