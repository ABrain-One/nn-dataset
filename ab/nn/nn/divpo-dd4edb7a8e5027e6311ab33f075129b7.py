import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._create_layer(32, 32, 1)
        self.layer2 = self._create_layer(32, 64, 1)
        self.layer3 = self._create_layer(64, 128, 2)
        self.layer4 = self._create_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(256 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, self.num_classes)
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

    def _create_layer(self, incoming: int, outgoing: int, blocks: int) -> nn.Sequential:
        layer = []
        layer += [
            nn.Conv2d(incoming, outgoing, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(outgoing),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1)
        ]
        for i in range(1, blocks):
            layer += [
                nn.Conv2d(outgoing, outgoing, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=1)
            ]
        self.prev_channel_num = outgoing
        return nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum, 
            dampening=0, 
            weight_decay=0, 
            nesterov=False
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            self.optimizer.step()