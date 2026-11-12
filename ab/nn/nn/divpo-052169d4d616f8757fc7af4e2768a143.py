import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        assert len(in_shape) == 4, "Input shape should be (batch, channels, height, width)"
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_layer(32, 32, 1)
        self.block2 = self._make_layer(32, 64, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(64 * 7 * 7, self.num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def _make_layer(self, in_planes: int, out_planes: int, blocks: int) -> nn.Sequential:
        seq_layers = []
        seq_layers.append(self._make_basic_block(in_planes, out_planes, blocks))
        return nn.Sequential(*seq_layers)

    def _make_basic_block(self, in_planes: int, out_planes: int, blocks: int) -> nn.Sequential:
        seq_layers = []
        seq_layers.append(self._make_basic_block_layer(in_planes, out_planes, blocks))
        seq_layers.append(nn.BatchNorm2d(out_planes))
        seq_layers.append(nn.ReLU())
        return nn.Sequential(*seq_layers)

    def _make_basic_block_layer(self, in_planes: int, out_planes: int, blocks: int) -> nn.Sequential:
        seq_layers = []
        seq_layers.append(nn.AvgPool2d(2, 2))
        seq_layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))
        for i in range(1, blocks):
            seq_layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1))
        return nn.Sequential(*seq_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x))
        x = self.block1(x)
        x = self.pool(self.block2(x))
        x = self.avgpool(x)
        x = x.view(-1, 64 * 7 * 7)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()