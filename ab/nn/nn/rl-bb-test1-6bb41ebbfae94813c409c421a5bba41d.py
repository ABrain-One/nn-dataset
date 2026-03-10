import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

# ==========================================
# 1. FIXED INFRASTRUCTURE (DO NOT MODIFY)
# ==========================================
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        kwargs = {"aux_logits": False} if "inception" in model.lower() else {}
        try:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
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

def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }

# ==========================================
# 2. DYNAMIC COMPONENTS (TO BE IMPLEMENTED)
# ==========================================

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
        nn.ReLU(inplace=True)
    )

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

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
            super().__init__()
            self.device = device
            self.use_amp = prm.get("use_amp", False)
            self._input_spec = in_shape[1], in_shape[2], in_shape[3]
            self.pattern = "Parallel_Triple"
            self.backbone_a = TorchVision(model='regnet_x_1_6gf', in_channels=self._input_spec[0])
            self.backbone_b = TorchVision(model='squeezenet1_0', in_channels=self._input_spec[0])
            self.features = nn.Sequential(
                FractalUnit(in_channels=self._input_spec[0], out_channels=64, num_columns=2, loc_drop_prob=0.15, dropout_prob=0.1),
                FractalUnit(in_channels=64, out_channels=128, num_columns=2, loc_drop_prob=0.15, dropout_prob=0.1)
            )
            self.infer_dimensions_dynamically(out_shape[0])
            self._scaler = make_scaler(enabled=self.use_amp)
        

    def infer_dimensions_dynamically(self, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            c, h, w = self._input_spec
            dummy = torch.zeros(1, c, h, w).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
            x_f = adaptive_pool_flatten(self.features(x)).flatten(1)
            x_a = adaptive_pool_flatten(self.backbone_a(x)).flatten(1)
            x_b = adaptive_pool_flatten(self.backbone_b(x)).flatten(1)
            fused = torch.cat([x_f, x_a, x_b], dim=1)
            if is_probing: return fused
            return self.classifier(fused)
        
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
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
                if not torch.isfinite(loss): continue
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
            if hasattr(train_iter, 'shutdown'): train_iter.shutdown()
            del train_iter
            gc.collect()
