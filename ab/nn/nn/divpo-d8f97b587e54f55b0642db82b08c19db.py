import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.block1(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.block2(x)
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

# Edge-friendly block to ensure fast latency on limited resources
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Calculate the number of parameters
def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def vaild_hyperparameters():
    return {'lr', 'momentum'}

def allowed_hyperparameters():
    return {'lr', 'momentum'}

def get_train_hyperparameters(default_prm):
    prm = {
        'lr': default_prm.get('lr', 0.01),
        'momentum': default_prm.get('momentum', 0.9),
    }
    return prm

def get_model():
    class Net(nn.Module):
        def __init__(self, in_shape, out_shape, prm, device):
            super().__init__()
            self.device = device
            in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
            num_classes = int(out_shape[0])
            self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.block1 = Block(32, 32)
            self.block2 = Block(32, 64)
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(64 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            x = self.stem(x)
            x = self.relu(x)
            x = self.block1(x)
            x = nn.functional.max_pool2d(x, 2, 2)
            x = self.block2(x)
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

    # Ensure the total parameters are under the limit
    net = Net(None, None, None, None)
    params = get_num_params(net)
    assert params <= 500000, f"Total parameters must be under 500000, but are {params}"
    return net

def supported_hyperparameters():
    return {'lr', 'momentum'}

def vaild_hyperparameters():
    return {'lr', 'momentum'}

def allowed_hyperparameters():
    return {'lr', 'momentum'}

def get_train_hyperparameters(default_prm):
    prm = {
        'lr': default_prm.get('lr', 0.01),
        'momentum': default_prm.get('momentum', 0.9),
    }
    return prm

def test_valid_hyperparameters_lr():
    prm = {'lr': 0.1}
    assert vaild_hyperparameters() == set(prm.keys())

def test_valid_hyperparameters_momentum():
    prm = {'momentum': 0.8}
    assert vaild_hyperparameters() == set(prm.keys())

def test_allowed_hyperparameters_lr():
    prm = {'lr': 0.1}
    assert allowed_hyperparameters