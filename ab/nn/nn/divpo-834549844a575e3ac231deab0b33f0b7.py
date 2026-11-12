import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * expand_ratio),
            nn.ReLU6(inplace=True),
            # Depthwise convolution
            nn.Conv2d(out_channels * expand_ratio, out_channels * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=out_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(out_channels * expand_ratio),
            # Pointwise convolution
            nn.Conv2d(out_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = 32
        self.num_classes = out_shape[0]
        self.dropout_rate = prm.get('dropout', 0.2)
        self.learning_rate = prm.get('lr', 1e-3)
        self.momentum = prm.get('momentum', 0.9)
        self.depth_multiplier = 1
        self.model_capacity = 1
        self.channel_multiplier = 1
        self.depthwise_multiplier = 1
        self.round_limit = 3

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(self.model_capacity * 16), self.num_classes)

        self.hyperparameters = {
            'lr': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': 0.05,
            'dropout': self.dropout_rate
        }

    def build_features(self):
        layers = []
        self.input_channels = self.in_channels
        self.input_resolution = self.image_size
        self.depth_round = self.depth_multiplier
        self.round_depth_limit = self.round_limit
        self.round_depth = self.round_depth_limit
        self.depth_round = self.depth_multiplier
        self.depth_limit = 35
        self.round_limit = 3
        self.depth_round = self.depth_multiplier
        self.depth_round = self.depth_multiplier
        self.depth_limit = 35
        self.round_depth_limit = self.round_limit
        self.depth_limit = 35
        self.round_depth = min(self.depth_limit, self.depth_round)
        self.round_depth = min(self.depth_limit, self.depth_round)
        self.depth_limit = 35
        self.round_depth = min(self.depth_limit, self.depth_round)

        self.block_args = self.make_block_args(
            kernel_size=3,
            block_args={
                'kernel_size': 3,
                'num_blocks': self.round_depth,
                'out_channels': int(self.model_capacity * 16),
                'expand_ratio': 1,
                'se_ratio': 0.25,
                'stride': 1,
                'act_layer': 'ReLU6',
            }
        )

        for i, block_arg in enumerate(self.block_args):
            layers.append(InvertedResidual(
                in_channels=self.input_channels,
                out_channels=block_arg['out_channels'],
                stride=block_arg['stride'],
                expand_ratio=block_arg['expand_ratio']
            ))

            self.input_channels = block_arg['out_channels']
            self.input_resolution = int(self.input_resolution / block_arg['stride'])

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

        return self.features

    def make_block_args(self, kernel_size, block_args):
        block_args = [block_args] * self.round_depth
        block_args[0]['stride'] = 2
        return block_args

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