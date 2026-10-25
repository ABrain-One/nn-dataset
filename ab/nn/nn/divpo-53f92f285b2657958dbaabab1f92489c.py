import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def _make_layer(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.stem = _make_layer(in_shape[1], 32, 3, padding=1)
        self.layer1 = _make_layer(32, 64, 3, stride=2, padding=1)
        self.layer2 = _make_layer(64, 128, 3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def get_complexity(self, input_size):
        parameters = 0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                parameters += layer.weight.element_size() * layer.weight.shape[0]
                if layer.bias is not None:
                    parameters += layer.bias.element_size() * layer.bias.shape[0]
            elif isinstance(layer, nn.BatchNorm2d):
                parameters += layer.weight.element_size() * layer.weight.shape[0]
        flops = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        return parameters, flops

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def get_flops(self, input_size):
        flops = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        return flops

    def get_parameter_size(self):
        return sum(p.numel() for p in self.parameters()) * p.element_size()

    def get_parameter_ratio(self):
        return self.get_parameter_size() / self.get_flops(input_size=(1, 3, 32, 32))

    def get_param_mult(self):
        return self.get_parameter_ratio() / self.get_parameter_ratio()

    def get_flops_mult(self):
        return self.get_flops(input_size=(1, 3, 32, 32)) / self.get_flops(input_size=(1, 3, 32, 32))

    def get_complexity_mult(self):
        return self.get_complexity(input_size=(1, 3, 32, 32)) / self.get_complexity(input_size=(1, 3, 32, 32))

    def get_param_mult_ratio(self):
        return self.get_param_mult() / self.get_flops_mult()

    def get_flops_mult_ratio(self):
        return self.get_flops_mult() / self.get_flops_mult()

    def get_complexity_mult_ratio(self):
        return self.get_complexity_mult() / self.get_complexity_mult()

    def get_param_flops_ratio(self):
        return self.get_param_mult() / self.get_flops_mult()

    def get_flops_complexity_ratio(self):
        return self.get_flops_mult() / self.get_complexity_mult()

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())

    def get_param_flops_complexity_ratio(self):
        return self.get_param_mult() / (self.get_flops_mult() * self.get_complexity_mult())