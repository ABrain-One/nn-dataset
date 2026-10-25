import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._last_channels = 128

        self._last_image_size = (32, 32)
        self._complete = False
   

    def _last_channel(self, c):
        return c <= self._last_channels

    def _next_channel(self, c):
        return c <= self._last_channels

    def _next_image_size(self, size):
        return size <= self._last_image_size

    def _verify_layer_dimensions(self, x, conv_layer, layer_name):
        if x.shape[1] != conv_layer.out_channels:
            raise ValueError(f'Layer {layer_name} has incorrect number of output channels.')
        if x.shape[2] != conv_layer.kernel_size[0] or x.shape[3] != conv_layer.kernel_size[1]:
            raise ValueError(f'Layer {layer_name} has incorrect kernel size.')

    def _conv_bn_relu(self, x, conv_layer, bn_layer, relu_layer):
        self._verify_layer_dimensions(x, conv_layer, 'Conv')
        x = conv_layer(x)
        self._verify_layer_dimensions(x, bn_layer, 'BatchNorm')
        x = bn_layer(x)
        self._verify_layer_dimensions(x, relu_layer, 'ReLU')
        return relu_layer(x)

    def _forward_until_last_channel(self, x, channel):
        if self._next_channel(channel):
            if self._last_image_size != (32, 32):
                raise ValueError('Image size does not match.')
            for module in self.features[:-1]:
                if isinstance(module, nn.Conv2d):
                    if module.out_channels != channel:
                        raise ValueError('Layer has incorrect number of output channels.')
                    if module.kernel_size[0] != 3 or module.kernel_size[1] != 3:
                        raise ValueError('Layer has incorrect kernel size.')
                    if module.in_channels != 32:
                        raise ValueError('Layer has incorrect number of input channels.')
                    x = module(x)
                else:
                    x = module(x)
            return x
        else:
            raise ValueError('Channel does not match.')

    def _forward_until_last_image_size(self, x, size):
        if self._next_image_size(size):
            for module in self.features[:-1]:
                if isinstance(module, nn.Conv2d):
                    if module.kernel_size[0] != size or module.kernel_size[1] != size:
                        raise ValueError('Layer has incorrect kernel size.')
                    x = module(x)
                else:
                    x = module(x)
            return x
        else:
            raise ValueError('Image size does not match.')

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
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
            self.optimizer.step()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def supported_hyperparameters(self):
        return ['lr', 'momentum']

    def filter_trainable_params(self, module, prefix=""):
        for name, param in module.named_parameters():
            if prefix + name not in self.optimizer.param_groups[0]["params"]:
                continue
            yield param

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.filter_trainable_params(self, prefix=""), max_norm=3)
            self.optimizer.step()

    def prune_params(self, params_by_layer):
        total_params = 0
        for layer, params in params_by_layer.items():
            if layer in self.state_dict():
                total_params += len(params)
        return total_params <= 500000

    def get_params_by_layer(self):
        params_by_layer = {}
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                params_by_layer[module] = list(module.parameters())
        return params_by_layer

    def get_flops(self, x):
        flops = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        return flops

    def get_params(self):
        total_params = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                total_params += sum(p.numel() for p in module.parameters())