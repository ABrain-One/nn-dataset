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
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_comp = None
        self._last_forwards = None
        self._last_activations = None
        self._last_layer_names = None
        self._last_layer_indices = None
        self._last_layer_types = None
        self._last_layer_params = None

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
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

    def hook_fn(self, module, input, output):
        if self._is_fwd_comp:
            self._last_forwards.append(output.cpu().detach())
        self._last_activations[module] = output.cpu().detach()

    def hook_last_layer_fn(self, module, input, output):
        self._last_layer_names[-1] = module
        self.hook_fn(module, input, output)

    def hook_all(self, fwd_comp=False):
        self._is_fwd_comp = fwd_comp
        if fwd_comp:
            self._last_forwards = []
        self._last_activations = {}
        self._last_layer_names = []
        self._last_layer_indices = []
        self._last_layer_types = []
        self._last_layer_params = []
        for name, module in self.named_modules():
            layer_type = type(module)
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                self._last_layer_names.append(name)
                self._last_layer_indices.append(int(name.split('.')[-1]))
                self._last_layer_types.append(layer_type)
                self._last_layer_params.append(list(module.parameters()))
                module.register_forward_hook(self.hook_fn)
                if fwd_comp:
                    module.register_forward_pre_hook(self.hook_fn)
        self.hook_last_layer_fn(self, None, None)

    def unhook_all(self):
        if self._last_layer_names:
            self._last_layer_names.pop()
            self._last_layer_indices.pop()
            self._last_layer_types.pop()
            self._last_layer_params.pop()
        self._is_fwd_comp = None
        self._last_forwards = None
        self._last_activations = None
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                module.unregister_forward_hook(self.hook_fn)
                if self._is_fwd_comp:
                    module.unregister_forward_pre_hook(self.hook_fn)
        self._last_layer_names = None
        self._last_layer_indices = None
        self._last_layer_types = None
        self._last_layer_params = None