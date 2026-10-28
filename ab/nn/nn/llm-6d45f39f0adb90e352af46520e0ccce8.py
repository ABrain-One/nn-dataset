import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

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
from torch.amp import autocast, GradScaler

def autocast_ctx(enabled=True):
    return autocast('cuda', enabled=enabled)

def make_scaler(enabled=True):
    return GradScaler('cuda', enabled=enabled)

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum', 'pretrained'}

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        out = self.dropout(out)
        return out

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.pattern = 'Optimized_Channel_Growth'
        self.use_amp = prm.get('use_amp', False)
        dropout_prob = float(prm.get('dropout', 0.2))
        backbone_weights = 'DEFAULT' if float(prm.get('pretrained', 1.0)) > 0 else None
        if len(in_shape) == 4:
            c_in, h_in, w_in = (in_shape[1], in_shape[2], in_shape[3])
        else:
            c_in, h_in, w_in = (in_shape[0], in_shape[1], in_shape[2])
        self._input_spec = (c_in, h_in, w_in)
        self.channels = [64, 128, 256, 512]
        self.features = nn.Sequential(ConvBlock(c_in, self.channels[0], stride=1, dropout_prob=dropout_prob), ConvBlock(self.channels[0], self.channels[0], stride=1, dropout_prob=dropout_prob), MaxPool2d(kernel_size=2, stride=2), ConvBlock(self.channels[0], self.channels[1], stride=2, dropout_prob=dropout_prob), ConvBlock(self.channels[1], self.channels[1], stride=1, dropout_prob=dropout_prob), ConvBlock(self.channels[1], self.channels[2], stride=2, dropout_prob=dropout_prob), ConvBlock(self.channels[2], self.channels[2], stride=1, dropout_prob=dropout_prob), ConvBlock(self.channels[2], self.channels[3], stride=2, dropout_prob=dropout_prob), ConvBlock(self.channels[3], self.channels[3], stride=1, dropout_prob=dropout_prob), MaxPool2d(kernel_size=2, stride=2), ConvBlock(self.channels[3], self.channels[3], stride=1, dropout_prob=dropout_prob))
        self.infer_dimensions_dynamically(out_shape[0])
        self._scaler = GradScaler('cuda', enabled=self.use_amp)

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

    def _init_params(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f'Expected 4D/5D input, got {tuple(x.shape)}')

    def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:
        x = self._norm4d(x)
        out = self.features(x)
        out = adaptive_pool_flatten(out)
        if is_probing:
            return out
        return self.classifier(out)

    def train_setup(self, prm):
        self.to(self.device)
        self._prm = prm
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        max_steps = int(self._prm.get('max_steps', 0)) if hasattr(self, '_prm') else 0
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
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
                if max_steps and batch_idx >= max_steps:
                    break
                batch_size = int(labels.size(0))
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((outputs.detach().argmax(dim=1) == labels).sum().item())
                total_examples += batch_size
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
        if total_examples == 0:
            return (0.0, None)
        train_accuracy = total_correct / total_examples
        train_loss = total_loss / total_examples
        return (train_accuracy, train_loss)