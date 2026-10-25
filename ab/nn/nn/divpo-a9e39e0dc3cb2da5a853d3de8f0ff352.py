import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.3)),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.3)),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_hyperparameter(hp_name, hp_value, supported_hp):
    if hp_name in supported_hp:
        if hp_name == 'lr' and not (isinstance(hp_value, (int, float)) and 0 < hp_value <= 1):
            raise ValueError(f'Invalid learning rate: {hp_value}. Must be a number in the range (0, 1].')
        if hp_name == 'momentum' and not (isinstance(hp_value, (int, float)) and 0 <= hp_value <= 1):
            raise ValueError(f'Invalid momentum: {hp_value}. Must be a number in the range [0, 1].')
        if hp_name == 'dropout' and not (isinstance(hp_value, (int, float)) and 0 <= hp_value < 1):
            raise ValueError(f'Invalid dropout rate: {hp_value}. Must be a number in the range [0, 1).')
        if hp_name == 'weight_decay' and not (isinstance(hp_value, (int, float)) and hp_value >= 0):
            raise ValueError(f'Invalid weight_decay: {hp_value}. Must be a non-negative number.')
    else:
        raise ValueError(f'Unsupported hyperparameter: {hp_name}')

    return hp_value

def validate_hyperparameters(prm, supported_hp):
    validated_hp = {}
    for hp_name, hp_value in prm.items():
        if hp_name in supported_hp:
            validated_hp[hp_name] = check_hyperparameter(hp_name, hp_value, supported_hp)
    return validated_hp

def get_parameter_groups(model):
    parameter_groups = {}
    for name, param in model.named_parameters():
        if 'weight_decay' in name:
            parameter_groups.setdefault('weight_decay', []).append(param)
        else:
            parameter_groups.setdefault('no_weight_decay', []).append(param)
    return parameter_groups

def get_parameter_groups_for_model(model, prm):
    prm = validate_hyperparameters(prm, supported_hyperparameters())
    return get_parameter_groups(model)

def get_parameter_groups_for_config(config, prm):
    prm = validate_hyperparameters(prm, supported_hyperparameters())
    return get_parameter_groups(config)

def get_parameter_groups_for_config_and_model(config, model, prm):
    prm = validate_hyperparameters(prm, supported_hyperparameters())
    return get_parameter_groups(config, model)

def get_parameter_groups_for_config_and_model_and_prm(config, model, prm):
    prm = validate_hyperparameters(prm, supported_hyperparameters())
    return get_parameter_groups(config, model, prm)

def get_parameter_groups_for_config_and_model_and_prm_and_something_else(config, model, prm, something_else):
    prm = validate_hyper