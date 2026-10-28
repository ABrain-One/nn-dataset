import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.dropout = prm['dropout']
        initial_channels = 64
        block1_channels = 128
        block2_channels = 256
        block3_channels = 512
        self.conv1 = nn.Conv2d(self.in_channels, initial_channels, kernel_size=7, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(initial_channels, block1_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(block1_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(block1_channels, block2_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(block2_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(block2_channels, block3_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(block3_channels)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(block3_channels, block3_channels, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(block3_channels)
        self.relu5 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc_dropout = nn.Dropout(p=self.dropout)
        self.fc1 = nn.Linear(block3_channels * 6 * 6, 256)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, self.out_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_dropout(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        return x