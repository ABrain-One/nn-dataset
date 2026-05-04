import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d

class TorchVision(nn.Module):

    def __init__(self, model: str, weights: str='DEFAULT', unwrap: bool=True, truncate: int=1, in_channels: int=3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        if hasattr(torchvision.models, 'get_model'):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if 'aux' in name.lower():
                    continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1) if x.ndim == 4 else x

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def drop_conv3x3_block(in_c, out_c, stride=1, padding=1, bias=False, dp=0.0):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, stride, padding, bias), nn.BatchNorm2d(out_c), nn.Dropout2d(dp) if dp else nn.Identity(), nn.SiLU(inplace=True))

class FractalBlock(nn.Module):

    def __init__(self, inc, oc, num_cols, loc_p, dp):
        super().__init__()
        depth = 2 ** max(num_cols - 1, 0)
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = nn.ModuleList()
            for j in range(num_cols):
                if (i + 1) % 2 ** j == 0:
                    c_in = inc if i == 0 and j == 0 else oc
                    layer.append(drop_conv3x3_block(c_in, oc, dp=dp))
            self.layers.append(layer)

    def forward(self, x):
        outs = [x] * len(self.layers[0])
        for lvl in self.layers:
            tmp = [blk(xi) for blk, xi in zip(lvl, outs)]
            merged = torch.stack(tmp, 0).mean(0)
            outs = [merged for _ in range(len(lvl))]
        return outs[0]

class FractalUnit(nn.Module):

    def __init__(self, inc, oc, cols, loc_p, dp):
        super().__init__()
        self.frac = FractalBlock(inc, oc, cols, loc_p, dp)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.frac(x))

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.backbone_a = TorchVision('regnet_x_400mf', 'IMAGENET1K_V1').to(device)
        self.backbone_b = TorchVision('efficientnet_b2', 'IMAGENET1K_V1').to(device)
        self.features = nn.Sequential()
        chans = [64, 128, 256]
        curr_ch = 3
        for i, oc in enumerate(chans):
            self.features.add_module(f'unit{i + 1}', FractalUnit(curr_ch if i == 0 else chans[i - 1], oc, 3, 0.15, prm['dropout']))
            curr_ch = oc
        self.infer_dimensions_dynamically(in_shape, out_shape[0])

    def infer_dimensions_dynamically(self, in_shape, num_clss):
        self.to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
            f_a = adaptive_pool_flatten(self.backbone_a(dummy))
            f_b = self.backbone_b(dummy)
            f_b = adaptive_pool_flatten(f_b)
            merged = torch.cat((f_a, f_b), 1)
            self.classifier = nn.Linear(merged.size(1), num_clss).to(self.device)
        self.train()

    def forward(self, x):
        xa = self.backbone_a(x)
        xa = adaptive_pool_flatten(xa)
        xb = self.backbone_b(x)
        xb = adaptive_pool_flatten(xb)
        merged = torch.cat((xa, xb), 1)
        return self.classifier(merged)

    def train_setup(self, prm):
        self.prm = prm
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']

        self.classifier.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD([param for param in self.parameters() if param.requires_grad], lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, data_loader):
        self.train()
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()