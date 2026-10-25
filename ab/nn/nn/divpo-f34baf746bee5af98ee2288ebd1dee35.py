import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, dropout=prm.get('dropout', 0.0))
        self.layer2 = self.build_layer(32, 64, 2, dropout=prm.get('dropout', 0.0))
        self.layer3 = self.build_layer(64, 128, 2, dropout=prm.get('dropout', 0.0))
        self.layer4 = self.build_layer(128, 256, 2, dropout=prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_prunable = None
        self._last_layer = None
        self._last_layer_name = None
        self._last_layer_dim = None
        self._last_layer_out_channels = None
        self._last_layer_kernel_size = None
        self._last_layer_stride = None
        self._last_layer_padding = None
        self._last_layer_dilation = None
        self._last_layer_transposed = None
        self._last_layer_output_padding = None
        self._last_layer_groups = None
        self._last_layer_bias = None
        self._last_layer_padding_mode = None
        self._last_layer_device = None
        self._last_layer_dtype = None
        self._last_layer_requires_grad = None
        self._last_layer_device_is_cpu = None
        self._last_layer_shape = None
        self._last_layer_out_shape = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(self.device)
        for p in self.parameters():
            p.requires_grad = True
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def build_layer(self, in_channels, out_channels, num_blocks, kernel_size=3, stride=1, padding=1, dropout=0.0):
        layers = []
        for i in range(num_blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        if dropout > 0:
            layers += [nn.Dropout(p=dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        for p in self.parameters():
            p.requires_grad = True
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
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def prune(self, layer, amount):
        if layer is None:
            return
        if self._is_prunable is None:
            self._is_prunable = [
                m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
            ]
            self._last_layer = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
            self._last_layer_name = list(self._last_layer.state_dict().keys())[0]
            self._last_layer_dim = self._last_layer.state_dict()[self._last_layer_name].shape
            self._last_layer_out_channels = self._last_layer.out_features
            self._last_layer_kernel_size = self._last_layer.kernel_size
            self._last_layer_stride = self._last_layer.stride
            self._last_layer_padding = self._last_layer.padding
            self._last_layer_dilation = self._last_layer.dilation
            self._last_layer_transposed = self._last_layer.transposed
            self._last_layer_output_padding = self._last_layer.output_padding
            self._last_layer_groups = self._last_layer.groups
            self._last_layer_bias = self._last_layer.bias is not None
            self._last_layer_padding_mode = self._last_layer.padding_mode
            self._last_layer_device = self._last_layer.weight.device
            self._last_layer_dtype = self._last_layer.weight.dtype
            self._last_layer_requires_grad = self._last_layer.weight.requires_grad
            self._last_layer_device_is_cpu = self._last_layer.weight.device.type == 'cpu'
            self._last_layer_shape = list(self._last_layer.weight.shape)
            self._last_layer_out_shape = list(self._last_layer.weight.shape[1:])
        self.prune_layer = layer
        self.prune_amount = amount
        self.prune_layers = []
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m is not layer:
                if m not in self._is_prunable:
                    continue
                self.prune_layers.append(m)
                prune.random_unstructured(
                    m,
                    name='weight',
                    amount=amount,
                    pruning_fn=prune.static.magnitude
                )
        if len(self.prune_layers) == 0:
            return
        self._last_layer = self.prune_layers[-1]
        self._last