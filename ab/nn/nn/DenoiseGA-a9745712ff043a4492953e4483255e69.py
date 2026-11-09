# name=DenoiseMegvii
# Megvii — winner of the MAI 2021 Real-Time Image Denoising Challenge, and the PSNR leader of
# Table 2 in Flepp/Ignatov CVPR2024 (38.05 dB / 0.884 SSIM / 56.8 ms). Ported layer-for-layer
# from the authors' released TensorFlow source (models/Megvii.py in
# rflepp/SplitterNet-Efficient-Mobile-Denoising-Models-CVPR2024), built as DYNUnet with
# enc_blocks=[1,1,1,1], dec_blocks=[1,1,1,1], num_filters=32 — the configuration their train.py
# uses. Parameter count is 3,563,907, matching the Keras build exactly.
#
# This is the model to beat on ACCURACY, as SplitterNet is the model to beat on LATENCY.
# It buys its +0.13 dB over SplitterNet with 4.9x the parameters and 2.05x the runtime, which
# is the inefficiency our challengers are meant to exploit. Like DenoiseSplitterNetX this is a
# reference port: do not tune it, fork it.
#
# Shape trace at 256x256 (channels after each stage):
#   head    3 -> 8 -> 32                                   256x256
#   enc1   32 -> 32 (k4 s2) -> 16 -> 64                    128x128    skip 32 @256
#   enc2   64 -> 64 (k4 s2) -> 32 -> 128                    64x64     skip 64 @128
#   enc3  128 -> 128 (k4 s2) -> 64 -> 256                   32x32     skip 128 @64
#   enc4  256 -> 256 (k4 s2) -> 128 -> 512                  16x16     skip 256 @32
#   dec1  512 -> 128 (+skip 256)                            32x32
#   dec2  128 -> 64  (+skip 128)                            64x64
#   dec3   64 -> 32  (+skip 64)                            128x128
#   dec4   32 -> 16  (+skip 32)                            256x256
#   tail   16 -> 3, plus global residual
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim

SLOPE = 0.3   # Keras LeakyReLU default; the authors never override it


def supported_hyperparameters():
    return {"lr"}


class EncoderBlk(nn.Module):
    """encoder_blk(): squeeze to f/2, then expand to 2f. Both convs are 3x3 'same'."""

    def __init__(self, c_in, f):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, f // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(f // 2, f * 2, 3, padding=1)
        self.act = nn.LeakyReLU(SLOPE)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))))


class DecoderBlk(nn.Module):
    """decoder_blk(): project the skip to `filters`, transpose-conv the input to match, add,
    fuse with a 3x3, then split the channels in half and convolve each half separately.

    Note the asymmetry: the first half gets a LeakyReLU after its conv and the second half does
    not. That is what their code does, so it is what this does."""

    def __init__(self, c_in, c_skip, f):
        super().__init__()
        self.skip_proj = nn.Conv2d(c_skip, f, 3, padding=1)
        self.up = nn.ConvTranspose2d(c_in, f, 2, stride=2)
        self.fuse = nn.Conv2d(f, f, 3, padding=1)
        self.half = f // 2
        self.conv_a = nn.Conv2d(self.half, self.half, 3, padding=1)
        self.conv_b = nn.Conv2d(self.half, self.half, 3, padding=1)
        self.act = nn.LeakyReLU(SLOPE)

    def forward(self, x, skip):
        net = self.act(self.fuse(self.up(x) + self.skip_proj(skip)))
        a = self.act(self.conv_a(net[:, :self.half]))
        b = self.conv_b(net[:, self.half:])          # deliberately no activation
        return torch.cat([a, b], dim=1)


class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 256, 256), out_shape=(3, 256, 256), prm={}, device="cuda"):
        super().__init__()
        self.device = device
        ch = in_shape[1]
        f = 32                                   # their num_filters

        self.head1 = nn.Conv2d(ch, 8, 3, padding=1)
        self.head2 = nn.Conv2d(8, f, 3, padding=1)
        self.act = nn.LeakyReLU(SLOPE)

        # encoder: 4 stages, each a k=4 s=2 downsample (zero-padded 1 on each side) + EncoderBlk
        self.downs = nn.ModuleList()
        self.encs = nn.ModuleList()
        self.skip_ch = []
        c = f
        nf = f
        for _ in range(4):
            self.skip_ch.append(c)
            self.downs.append(nn.Conv2d(c, nf, 4, stride=2))
            self.encs.append(EncoderBlk(nf, nf))
            c = nf * 2
            nf = nf * 2

        # decoder: mirrors the encoder, halving `filters` at every step
        self.decs = nn.ModuleList()
        nf = nf // 2                              # undo the final doubling
        for j in range(4):
            nf = nf // 2
            self.decs.append(DecoderBlk(c, self.skip_ch[3 - j], nf))
            c = nf

        self.tail = nn.Conv2d(c, ch, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        h = self.act(self.head2(self.act(self.head1(x))))
        skips = []
        for down, enc in zip(self.downs, self.encs):
            skips.append(h)
            h = enc(down(Fn.pad(h, (1, 1, 1, 1))))   # CONSTANT (zero) pad, matching their tf.pad
        for j, dec in enumerate(self.decs):
            h = dec(h, skips[3 - j])
        return self.tail(h) + x                      # global residual, unclamped as in their code

    def train_setup(self, prm):
        # Same recipe as DenoiseSplitterNetX so the head-to-head is like-for-like: the paper's
        # MSE loss and cosine 1e-4 -> 7e-6 (Sec 3.2), applied to every model it compares.
        self.loss_kind = prm.get("loss", "mse")
        lr = prm.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(int(prm.get("epoch", 15)), 1), eta_min=lr * 0.07)

    def psnr_loss(self, preds, target):
        mse = torch.mean((preds - target) ** 2, dim=[1, 2, 3])
        return -torch.mean(10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)))

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        for noisy, clean in train_data:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            self.optimizer.zero_grad()
            preds = self(noisy)
            loss = self.psnr_loss(preds, clean) if self.loss_kind == "psnr" \
                else self.criterion_mse(preds, clean)
            if not torch.isfinite(loss):
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        self.scheduler.step()
        return total_loss / max(count, 1)
