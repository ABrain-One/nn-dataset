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
    if x.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class FractalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % 2 ** j == 0:
                    in_ch_ij = in_channels if i + 1 == 2 ** j else out_channels
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            temp_outs = []
            for blk, ip in zip(level_block, outs):
                temp_outs.append(blk(ip))
            merged = torch.stack(temp_outs, dim=0).mean(dim=0)
            for idx in range(len(level_block)):
                outs[idx] = merged
        return outs[0]

class FractalUnit(nn.Module):

    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.fractal_block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.maxpool(self.fractal_block(x))

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.dropout_prob = float(prm.get('dropout', 0.1))
        self.backbone_a = TorchVision('efficientnet_v2_s', in_channels=3).to(device)
        self.backbone_b = TorchVision('regnet_y_1_6gf', in_channels=3).to(device)
        self.features = nn.Sequential()
        channels = [64, 128]
        curr_ch = 3
        for i, out_ch in enumerate(channels):
            self.features.add_module(f'unit{i + 1}', FractalUnit(curr_ch, out_ch, 2, 0.15, self.dropout_prob))
            curr_ch = out_ch
        self.classifier = None
        self.infer_dimensions_dynamically(in_shape, out_shape[0])

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape) == 4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
            features_a = self.backbone_a(dummy)
            features_b = self.backbone_b(dummy)
            merged = torch.cat((features_a, features_b), dim=1)
            merged = adaptive_pool_flatten(merged)
            dim = merged.size(1)
            self.classifier = nn.Linear(dim, num_classes)
        self.train()

    def forward(self, x, is_probing=False):
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.view(-1, x.size(0), 1, 1)
        x = self._preprocess(x)
        features_a = self.backbone_a(x)
        features_b = self.backbone_b(x)
        merged = torch.cat((features_a, features_b), dim=1)
        merged = adaptive_pool_flatten(merged)
        if is_probing:
            return merged
        return self.classifier(merged)

    def _preprocess(self, x):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.2023, 0.1994, 0.201]).view(-1, 1, 1).to(self.device)
        return (x - mean) / std

    def train_setup(self, prm):
        self.prm = prm
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']

        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_loader):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = (data.to(self.device), target.to(self.device))
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()