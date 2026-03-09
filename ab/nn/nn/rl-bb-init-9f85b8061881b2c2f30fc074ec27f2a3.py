import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

# -------------------------------------------------
# Wrapper & Helpers
# -------------------------------------------------
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if "aux" in name.lower(): continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    if x.ndim == 4: return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3: return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

# -------------------------------------------------
# AMP / Hyperparams
# -------------------------------------------------
from torch.amp import autocast, GradScaler
def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }

# -------------------------------------------------
# Generated Conv Block (Crucial Fix)
# -------------------------------------------------
def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
        nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
        nn.ReLU(inplace=True)
    )

# -------------------------------------------------
# Fractal Components (Using drop_conv3x3_block)
# -------------------------------------------------
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint_per_subblock = False

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

# -------------------------------------------------
# Net (Dynamic Fusion Architecture)
# -------------------------------------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = "Ensemble_Backbones_to_Fractal"
        self.use_amp = prm.get("use_amp", False)
        dropout_prob = float(prm.get('dropout', 0.1))

        # --- Filled by Generator ---
        self.backbone_a = TorchVision("resnet18", in_channels=3).to(device)

        self.features = nn.Sequential()
        curr_ch = 1280
        channels = [64 * (2 ** i) for i in range(1)]
        for i, out_ch in enumerate(channels):
            self.features.add_module(f"unit{i+1}", FractalUnit(curr_ch, out_ch, 3, 0.15, dropout_prob))
            curr_ch = out_ch

        self.backbone_b = TorchVision("convnext_tiny", in_channels=3).to(device)

        self.infer_dimensions_dynamically(in_shape, out_shape[0])
        self._scaler = GradScaler("cuda", enabled=self.use_amp)

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape)==4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]

        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    def _init_params(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None: nn.init.zeros_(m.bias)

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input, got {tuple(x.shape)}")

    
    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        x = self._norm4d(x).to(self.device)
        f_a = adaptive_pool_flatten(self.backbone_a(x))
        f_b = adaptive_pool_flatten(self.backbone_b(x))
        mid = torch.cat([f_a, f_b], dim=1)
        mid_4d = mid.unsqueeze(-1).unsqueeze(-1)
        mid_img = torch.nn.functional.interpolate(mid_4d, size=(14,14), mode='nearest')
        
        fused = adaptive_pool_flatten(self.features(mid_img))
        if is_probing: return fused
        return self.classifier(fused)
    

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)

                if not torch.isfinite(loss):
                    continue

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'):
                train_iter.shutdown()
            del train_iter
            gc.collect()