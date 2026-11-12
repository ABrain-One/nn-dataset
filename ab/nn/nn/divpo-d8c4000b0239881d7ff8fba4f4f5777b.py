import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expansion=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.expansion = expansion

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64, stride=2, expansion=2)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64, stride=2, expansion=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 32
        for x in cfg:
            out_channels = x if isinstance(x, int) else x[0]
            layers.append(Block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_layers(self, in_channels: int, layer_params: list, layer_blocks: list):
        layers = []
        for _ in range(layer_params[0]):
            layers.append(Block(in_channels, layer_params[1], layer_params[2], layer_params[3], layer_params[4], layer_blocks[0][1]))
            in_channels = layer_params[1] * layer_blocks[0][1]
        return layers

    def __repr__(self):
        return f"{self.__class__.__name__}(blocks={self._layers_repr})"

    def _layers_repr(self):
        return [f"{block.__class__.__name__}{block._expansion_repr}({block.conv1.in_channels}, {block.conv1.out_channels})" for block in self._blocks_in_order]

    def _blocks_in_order(self):
        return sorted(self._blocks_by_order.values(), key=lambda b: b.idx)

    def _blocks_by_order(self):
        return {block.idx: block for block in self._blocks_in_order()}

    def _check_cfg_length(self, cfg):
        if len(cfg) != len(self._blocks_in_order):
            raise ValueError(f"CFG length {len(cfg)} must match # of blocks {len(self._blocks_in_order)}")

    def _cfg_from_blocks(self, cfg):
        self._check_cfg_length(cfg)
        return [cfg[block.idx] for block in self._blocks_in_order()]

    def _cfg_from_params(self, cfg):
        self._check_cfg_length(cfg)
        return [(cfg[block.idx], block.out_channels, block.stride, block.padding, block.expansion) for block in self._blocks_in_order()]

    def _make_layers(self, cfg):
        layers = []
        in_channels = 32
        for x in cfg:
            out_channels = x if isinstance(x, int) else x[0]
            layers.append(Block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_layers(self, in_channels: int, layer_params: list, layer_blocks: list):
        layers = []
        for _ in range(layer_params[0]):
            layers.append(Block(in_channels, layer_params[1], layer_params[2], layer_params[3], layer_params[4], layer_blocks[0][1]))
            in_channels = layer_params[1] * layer_blocks[0][1]
        return layers

    def _make_layers(self, cfg):
        layers = []
        in_channels = 32
        for x in cfg:
            out_channels = x if isinstance(x, int) else x[0]
            layers.append(Block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_layers(self, in_channels: int, layer_params: list, layer_blocks: list):
        layers = []
        for _ in range(layer_params[0]):
            layers.append(Block(in_channels, layer_params[1], layer_params[2], layer_params[3], layer_params[4], layer_blocks[0][1]))
            in_channels = layer_params[1] * layer_blocks[0][1]
        return layers

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64, stride=2, expansion=2)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64, stride=2, expansion=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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