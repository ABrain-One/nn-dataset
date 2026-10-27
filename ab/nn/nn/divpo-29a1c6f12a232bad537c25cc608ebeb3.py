import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = conv_bn(self.in_channels, 32, 2)
        self.conv2 = conv_bn(32, 64, 2)
        self.conv3 = conv_bn(64, 128, 2)
        self.conv4 = conv_bn(128, 128, 1)
        self.conv5 = conv_bn(128, 128, 1)
        self.conv6 = conv_bn(128, 128, 1)
        self.conv7 = conv_1x1_bn(128, 128)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, self.num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

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

bounds = [128, 128, 128, 64, 32, 32, 32, 128]

def get_model_complexity_info(model, input_size):
    input_size = [1, 3, 32, 32]
    dummy_input = torch.randn(input_size)
    flops = torchvision.models.utils._log_api_usage_once(model._get_name())
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    return flops, params

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(bounds[0], self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, bounds[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bounds[0]),
            nn.ReLU(inplace=True),
        ]

        for i in range(len(bounds) - 1):
            layers += [
                nn.Conv2d(bounds[i], bounds[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(bounds[i + 1]),
                nn.ReLU(inplace=True),
            ]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

def supported_hyperparameters():
    return ['lr', 'momentum']

def check_shape(x, expected):
    if x.shape[1] != 3: raise AssertionError('Expected 3 channels')
    if x.shape[2] != 32: raise AssertionError('Expected image size of 32x32')
    if x.shape[3] != 32: raise AssertionError('Expected image size of 32x32')

def export_model(config):
    device = torch.device('cpu')
    model = Net(
        in_shape=(3, 32, 32),
        out_shape=(10,),
        prm={'lr': 0.01, 'momentum': 0.9},
        device=device,
    )
    return model