import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build__layer_max(128, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._last_layer_name = 'classifier'
        self._last_dim = out_shape[0]

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def build__layer_max(self, in_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers += [
                nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def cross_entropy_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.cross_entropy_loss(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def get_last_layer_name(self):
        return self._last_layer_name

    def get_last_layer_dim(self):
        return self._last_dim

    def get_flops(self, x):
        from torchvision.models.utils import conv_flops_counter_fn
        x = conv_flops_counter_fn(x, self._last_layer_name)
        return x

    def get_params_number(self):
        total_params = 0
        for layer in self.parameters():
            if layer.requires_grad:
                params = layer.numel()
                total_params += params
        return total_params

    def get_memory_usage(self):
        total_memory = 0
        for layer in self.parameters():
            if layer.requires_grad:
                memory = layer.element_size() * layer.numel()
                total_memory += memory
        return total_memory

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def get_last_layer_name(self):
        return self._last_layer_name

    def get_last_layer_dim(self):
        return self._last_dim

    def get_flops(self, x):
        from torchvision.models.utils import conv_flops_counter_fn
        x = conv_flops_counter_fn(x, self._last_layer_name)
        return x

    def get_params_number(self):
        total_params = 0
        for layer in self.parameters():
            if layer.requires_grad:
                params = layer.numel()
                total_params += params
        return total_params

    def get_memory_usage(self):
        total_memory = 0
        for layer in self.parameters():
            if layer.requires_grad:
                memory = layer.element_size() * layer.numel()
                total_memory += memory
        return total_memory

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}