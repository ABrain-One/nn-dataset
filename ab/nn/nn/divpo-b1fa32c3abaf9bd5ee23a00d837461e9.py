import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 8, 3, stride=1, padding=1)  # type: ignore
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._block(8, 16, kernel_size=3, stride=1, padding=1, num_blocks=2)
        self.layer2 = self._block(16, 32, kernel_size=3, stride=1, padding=1, num_blocks=3)
        self.layer3 = self._block(32, 64, kernel_size=3, stride=1, padding=1, num_blocks=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.3))  # type: ignore
        self.fc = nn.Linear(64, out_shape[0])

    def _block(self, input_channels: int, output_channels: int, kernel_size: int,
              stride: int, padding: int, num_blocks: int) -> nn.Sequential:
        block = nn.Sequential()
        for idx in range(num_blocks):
            if idx == 0:
                block.add_module(f"conv{idx}", nn.Conv2d(input_channels, output_channels, kernel_size,
                                                       stride, padding))
            else:
                block.add_module(f"conv{idx}", nn.Conv2d(output_channels, output_channels, kernel_size,
                                                        stride, padding))
            block.add_module(f"relu{idx}", nn.ReLU())
            block.add_module(f"bn{idx}", nn.BatchNorm2d(output_channels))
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
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