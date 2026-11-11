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
        self.layer1 = self._make_layer(Block, 32, 32, 1)
        self.layer2 = self._make_layer(Block, 32, 64, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, block, in_planes, out_planes, blocks, downsample=False):
        layers = []
        for i in range(blocks):
            if i == 0 and downsample:
                layers.append(block(in_planes, out_planes, downsample=True))
            else:
                layers.append(block(in_planes, out_planes, downsample=False))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
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

# Optional: helper blocks AFTER Net definition
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample_layer(x)
        out += residual
        return self.relu(out)

    def extra_repr(self):
        return f'in_channels={self.conv1.in_channels}, out_channels={self.conv1.out_channels}, downsample={self.downsample}'


def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_hyperparameter(hyperparameter):
    return hyperparameter in supported_hyperparameters()

def get_class_by_name(name):
    if name == "Net":
        return Net
    raise Exception("Unknown class name: " + name)

def build_network_by_config(network_config):
    if not isinstance(network_config, dict) or "name" not in network_config:
        raise Exception("Network config is invalid")

    cls = get_class_by_name(network_config["name"])

    hyperparameters = network_config.get("hyperparameters", {})
    for hyperparam_name, hyperparam_value in hyperparameters.items():
        if not check_hyperparameter(hyperparam_name):
            raise Exception("Hyperameter is not supported by the model: " + hyperparam_name)

    networks = []
    for layer_config in network_config["layers"]:
        if "children" in layer_config:
            child_config = layer_config["children"].copy()
            child_config["hyperparameters"] = hyperparameters
            networks.append(build_network_by_config(child_config))
        else:
            net = cls(in_channels=layer_config["out_channels"],
                      out_channels=layer_config["out_channels"],
                      kernel_size=layer_config["kernel_size"],
                      stride=layer_config["stride"],
                      padding=layer_config["padding"],
                      groups=layer_config["groups"],
                      bias=False if "batch_norm" in layer_config else True)

            if "batch_norm" in layer_config:
                net.add_module("batch_norm",
                               nn.BatchNorm2d(layer_config["out_channels"], momentum=hyperparameters.get("momentum", 0.1)))
            networks.append(net)

    previous_channels = layer_config["out_channels"] if "out_channels" in layer_config else layer_config["in_channels"]
    for net in networks:
        if net.out_channels == previous_channels:
            previous_channels = net.in_channels
        else:
            previous_channels = net.out_channels

    return Net(networks, previous_channels, hyperparameters)