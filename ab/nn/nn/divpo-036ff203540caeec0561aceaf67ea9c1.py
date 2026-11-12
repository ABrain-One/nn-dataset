import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._create_layer(32, 32, 1)
        self.layer2 = self._create_layer(32, 64, 1)
        self.layer3 = self._create_layer(64, 128, 2)
        self.layer4 = self._create_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(256 * 7 * 7, self.num_classes)

    def _create_layer(self, inc: int, outc: int, blocks: int):
        layer = nn.Sequential()
        layer.add_module(name=f"conv1", module=nn.Conv2d(inc, outc, kernel_size=3,
                                                            padding=1, bias=False))
        layer.add_module(name=f"bn1", module=nn.BatchNorm2d(outc))
        layer.add_module(name=f"relu1", module=nn.ReLU(inplace=True))
        layer.add_module(name=f"maxpool1", module=nn.MaxPool2d(kernel_size=1))
        for block_idx in range(1, blocks):
            layer.add_module(name=f"conv{block_idx*2+1}", module=nn.Conv2d(outc, outc,
                                                                                  kernel_size=3,
                                                                                  padding=1,
                                                                                  bias=False))
            layer.add_module(name=f"bn{block_idx*2+1}", module=nn.BatchNorm2d(outc))
            layer.add_module(name=f"relu{block_idx*2+1}", module=nn.ReLU(inplace=True))
            layer.add_module(name=f"maxpool{block_idx*2+1}",module=nn.MaxPool2d(kernel_size=1))
        self._initialize_weights(layer)
        return layer

    def _initialize_weights(self, module):
        for name, obj in module.named_modules():
            if isinstance(obj, nn.Conv2d):
                import numpy as np
                nn.init.kaiming_normal_(obj.weight.data, mode="fan_out", nonlinearity="relu")
                if obj.bias is not None:
                    obj.bias.data.zero_()
            elif isinstance(obj, nn.Bilinear):
                nn.init.kaiming_normal_(obj.weight.data, mode='fan_out')
                nn.init.constant_(obj.bias.data, 0)
            elif isinstance(obj, nn.BatchNorm2d):
                nn.init.constant_(obj.weight.data, 1)
                nn.init.constant_(obj.bias.data, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self._forward_block(x, self.layer1)
        x = self._forward_block(x, self.layer2)
        x = self._forward_block(x, self.layer3)
        x = self._forward_block(x, self.layer4)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def _forward_block(self, x, layer):
        through = x
        for mod in layer.modules():
            if isinstance(mod, nn.Conv2d):
                x = mod(x)
            elif isinstance(mod, nn.BatchNorm2d):
                x = mod(x)
            elif isinstance(mod, nn.ReLU):
                x = mod(x)
            elif isinstance(mod, nn.MaxPool2d):
                x = mod(x)
            elif isinstance(mod, nn.AdaptiveAvgPool2d):
                x = mod(x)
            elif hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                x = mod(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()