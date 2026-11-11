import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxnolayer = nn.MaxPool2d(2)
        self.layer1 = self.build_layer(32, 32, 32)
        self.layer2 = self.build_layer(32, 64, 64)
        self.layer3 = self.build_layer(64, 128, 16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.linear = nn.Linear(128, self.num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxnolayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def build_layer(self, max_width, end_width, block_num):
        layers = []
        strdie = 1
        layers += [
            nn.Conv2d(max_width, end_width, kernel_size=3,
                      stride=strdie, padding=1, bias=False),
            nn.BatchNorm2d(end_width),
            nn.ReLU(inplace=True)]
        self.width = end_width
        self.block_num = block_num
        return nn.Sequential(*layers)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()