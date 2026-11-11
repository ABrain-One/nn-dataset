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
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, 3, 1)
        self.layer2 = self._make_layer(64, 128, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
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

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_requirements(self, batch_size=1):
        dummy_input = torch.randn(batch_size, *in_shape).to(self.device)
        return sum(t.numel() * t.element_size() for t in self.forward(dummy_input))

# Dummy in_shape to adhere to the standard contract (C, H, W)
in_shape = (3, 32, 32)
out_shape = (10,)
prm = {'lr': 0.01, 'momentum': 0.9}
device = torch.device('cpu')

# Generate a random 500000-param model within the memory budget (for size limit)
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _calculate_channel_mult(channel_mult, num_layers):
    return [channel_mult] * num_layers

def _generate_model(channel_mult=1.0, num_blocks=(3, 3, 9, 3), num_layers=5, width_mult=1.0):
    in_channels = in_shape[1]
    num_classes = out_shape[0]
    block_args = [
        dict(kernel_size=3, num_repeat=n, input_channels=in_channels,
             hidden_channels=_make_divisible(in_channels * width_mult, 8),
             out_channels=_make_divisible(in_channels * width_mult, 8))
        for n, in_channels in zip(num_blocks, _calculate_channel_mult(channel_mult, num_layers))
    ]
    block_args = [
        dict(kernel_size=3, num_repeat=num_blocks[0], input_channels=in_channels,
             hidden_channels=_make_divisible(in_channels * width_mult, 8),
             out_channels=_make_divisible(in_channels * width_mult, 8))
    ]
    block_args = block_args[:num_layers]

    layers = []
    for block_arg in block_args:
        layers.extend([
            nn.Conv2d(block_arg['input_channels'], block_arg['hidden_channels'],
                      kernel_size=block_arg['kernel_size'], padding=1, bias=False),
            nn.BatchNorm2d(block_arg['hidden_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(block_arg['hidden_channels'], block_arg['out_channels'],
                      kernel_size=block_arg['kernel_size'], padding=1, bias=False),
            nn.BatchNorm2d(block_arg['out_channels']),
            nn.ReLU(inplace=True)
        ])

    # Calculate the total number of channels
    total_channels = block_args[-1]['out_channels']
    # Calculate the total number of blocks
    total_blocks = len(block_args)
    # Calculate the total number of layers
    total_layers = total_blocks * 3

    # Create the model
    model = nn.Sequential(
        nn.Conv2d(in_channels, total_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(total_channels),
        nn.ReLU(inplace=True)
    )
    model.add_module('layer1', nn.Sequential(*layers[:total_layers]))
    model.add_module('layer2', nn.Sequential(*layers[total_layers:total_layers * 2]))
    model.add_module('layer3', nn.Sequential(*layers[total_layers * 2:total_layers * 3]))
    model.add_module('avgpool', nn.AdaptiveAvgPool2d((6, 6)))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc1', nn.Linear(total_channels * 6 * 6, 4096))
    model.add_module('relu1', nn.ReLU(inplace=True))
    model.add_module('fc2', nn.Linear(4096, num_classes))

    return model

Net = Net