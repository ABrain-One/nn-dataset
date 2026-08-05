
def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

from typing import Callable, List, Optional
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
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        layers.extend([Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation_layer=nn.ReLU6), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SqueezeExcitation(nn.Module):

    def __init__(self, input_c: int, squeeze_factor: int=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = nn.functional.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = nn.functional.hardsigmoid(scale, inplace=True)
        return scale * x

class SEInvertedResidual(InvertedResidual):

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super(SEInvertedResidual, self).__init__(inp, oup, stride, expand_ratio, norm_layer)
        self.se = SqueezeExcitation(oup)

def supported_hyperparameters():
    return ('batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform')

class Net(nn.Module):

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._transform = prm['transform']
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.dropout = nn.Dropout(prm['dropout'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def infer_dimensions_dynamically(self):
        dummy = torch.zeros(1, self.in_shape[1], self.in_shape[2], self.in_shape[3]).to(self.device)
        x = self.features(dummy)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        self.inferred_features_dim = x.size(1)

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        num_classes: int = out_shape[0]
        width_mult: float = 1.0
        inverted_residual_setting: Optional[List[List[int]]] = None
        round_nearest: int = 8
        block: Optional[Callable[..., nn.Module]] = None
        norm_layer: Optional[Callable[..., nn.Module]] = None
        if block is None:
            block = SEInvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 16
        last_channel = 640
        if inverted_residual_setting is None:
            inverted_residual_setting = [[3, 16, 1, 1], [3, 24, 2, 2], [3, 32, 3, 2], [3, 64, 4, 2], [3, 96, 3, 1], [3, 160, 3, 2], [3, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(f'inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}')
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [Conv2dNormActivation(in_shape[1], input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(p=prm['dropout']), nn.Linear(self.last_channel, num_classes))
        self.to(self.device)
        self.infer_dimensions_dynamically()
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)