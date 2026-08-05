# promoted name=DenoiseELUCtxXDenoiseUnet9F+sXDenoiseLin~c77e5b+m psnr50=36.014 macs=3330959872
# name=DenoiseELUCtxXDenoiseUnet9F+sXDenoiseLin~c77e5b+m parent=DenoiseELUCtxXDenoiseUnet9F+sXDenoiseLin~c77e5b checksum=3fb2f2e558e6ef1e
# psnr_db=33.8478 eff_db=33.8838 macs=3330959872 params=636323
import torch
import torch.nn as nn
import torch.optim as optim

class ELUAvgBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        c = channels
        self.norm1 = Xb_Xb__Norm2d(c)
        self.proj_in = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.proj_out = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)
        self.norm2 = Xb_Xb__Norm2d(c)
        self.ffn_up = nn.Conv2d(c, c * 4, 1)
        self.ffn_dn = nn.Conv2d(c * 2, c, 1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)

    def forward(self, inp):
        x = self.norm1(inp)
        xp = self.proj_in(x)
        x1, x2 = xp.chunk(2, dim=1)
        x1 = self.dw(x1)
        x = x1 * x2
        x = x * self.sca(x)
        x = self.proj_out(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.ffn_up(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.ffn_dn(x)
        return y + x * self.gamma

class Xb_Xb__Norm2d(nn.Module):
    """Mobile-GPU-clean norm: BatchNorm2d folds into the preceding conv on the
    TFLite GPU delegate. LayerNorm (the NAFNet default) + the permute round-trip
    are only partially supported and force CPU fallbacks."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.norm(x)

class Net(nn.Module):
    """ELUCtx — introduces op types absent from the seed pool; crossover-capable channel-preserving block."""

    def __init__(self, in_shape=(1, 3, 512, 512), out_shape=None, prm={}, device='cuda'):
        super().__init__()
        self.device = device
        Block = ELUAvgBlock
        self.in_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.eb0 = Block(16)
        self.down0 = nn.Conv2d(16, 32, 7, stride=2, padding=3)
        self.eb1 = Block(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.eb2 = Block(64)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bottleneck = Block(128)
        self.pre2 = nn.Conv2d(128, 512, 1)
        self.shuf2 = nn.PixelShuffle(2)
        self.up2 = nn.Conv2d(192, 64, 3, padding=1)
        self.db2 = Block(64)
        self.pre1 = nn.Conv2d(64, 256, 1)
        self.shuf1 = nn.PixelShuffle(2)
        self.up1 = nn.Conv2d(96, 32, 3, padding=1)
        self.db1 = Block(32)
        self.pre0 = nn.Conv2d(32, 128, 1)
        self.shuf0 = nn.PixelShuffle(2)
        self.up0 = nn.Conv2d(48, 16, 3, padding=1)
        self.db0 = Block(16)
        self.out_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.train_setup(prm)
        self.to(device)

    def forward(self, x):
        identity = x
        e0 = self.in_conv(x)
        e0 = self.eb0(e0)
        e1 = self.down0(e0)
        e1 = self.eb1(e1)
        e2 = self.down1(e1)
        e2 = self.eb2(e2)
        bt = self.down2(e2)
        bt = self.bottleneck(bt)
        d2 = self.db2(self.up2(torch.cat([self.shuf2(self.pre2(bt)), e2], 1)))
        d1 = self.db1(self.up1(torch.cat([self.shuf1(self.pre1(d2)), e1], 1)))
        d0 = self.db0(self.up0(torch.cat([self.shuf0(self.pre0(d1)), e0], 1)))
        return torch.clamp(self.out_conv(d0) + identity, 0.0, 1.0)

    def train_setup(self, prm):
        lr = prm.get('lr', 0.0001)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0.0
        count = 0
        last_noisy = None
        bn_layers = [m for m in self.modules() if isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]
        for noisy, clean in train_data:
            noisy, clean = (noisy.to(self.device), clean.to(self.device))
            last_noisy = noisy
            self.optimizer.zero_grad()
            bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                         m.num_batches_tracked.clone()) for m in bn_layers]
            preds = self(noisy)
            loss_gt = self.criterion_mse(preds, clean)
            loss = loss_gt * 1000 + self.criterion_l1(preds, clean) * 50
            bad = not torch.isfinite(loss)
            if not bad and bn_layers:
                bad = not all(torch.isfinite(m.running_mean).all()
                              and torch.isfinite(m.running_var).all() for m in bn_layers)
            if bad:
                # BN buffers were updated by this batch's forward BEFORE the
                # check — restore them or eval is poisoned while train stays ok
                for m, (rm, rv, nb) in zip(bn_layers, bn_state):
                    m.running_mean.copy_(rm)
                    m.running_var.copy_(rv)
                    m.num_batches_tracked.copy_(nb)
                continue
            if not torch.isfinite(loss): continue  # NaN-guard (yield>90%)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss_gt.item()
            count += 1
        self.scheduler.step()
        if bn_layers and count and last_noisy is not None:
            self.eval()
            with torch.no_grad():
                probe = self(last_noisy)
            if not torch.isfinite(probe).all():
                print('[bn-heal] eval-mode output non-finite; recalibrating BN '
                      'running stats', flush=True)
                saved_mom = [m.momentum for m in bn_layers]
                for m in bn_layers:
                    m.reset_running_stats()
                    m.momentum = None
                self.train()
                with torch.no_grad():
                    for _i, (_n, _c) in enumerate(train_data):
                        self(_n.to(self.device))
                        if _i >= 9:
                            break
                for m, mo in zip(bn_layers, saved_mom):
                    m.momentum = mo
                self.eval()
                with torch.no_grad():
                    probe = self(last_noisy)
                print(f'[bn-heal] post-recalibration finite='
                      f'{torch.isfinite(probe).all().item()}', flush=True)
            self.train()
        return total_loss / max(count, 1)