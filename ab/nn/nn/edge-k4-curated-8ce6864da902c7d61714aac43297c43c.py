from typing import Any, Callable, Optional
import torch
from torch import nn, Tensor
from torchvision.models._utils import _make_divisible
from torchvision.ops.misc import Conv2dNormActivation

class InvertedResidual(nn.Module):

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        layers.extend([Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, padding=1, norm_layer=norm_layer, activation_layer=nn.ReLU6), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class Net(nn.Module):

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape, out_shape, prm, device) -> None:
        super().__init__()
        self.device = device
        num_classes = out_shape[0]
        width_mult = 1.0
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d
        self.dropout_rate = prm['dropout']
        input_channels = 32
        last_channel = 1280
        input_channels = _make_divisible(input_channels * width_mult, 8)
        features = [Conv2dNormActivation(in_shape[1], input_channels, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)]
        inverted_residual_settings = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
        for expand, output, repeats, stride in inverted_residual_settings:
            output_c = _make_divisible(output * width_mult, 8)
            for i in range(repeats):
                stride = stride if i == 0 else 1
                features.append(block(input_channels, output_c, stride, expand, norm_layer))
                input_channels = output_c
        features.append(Conv2dNormActivation(input_channels, last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(p=self.dropout_rate), nn.Linear(last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x