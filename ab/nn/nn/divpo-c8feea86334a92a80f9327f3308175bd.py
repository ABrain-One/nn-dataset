import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(prm.get('dropout', 0.2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(160)
        self.conv6 = nn.Conv2d(160, 160, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(160)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, self.num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.xavier_uniform_(self.conv2.weight.data)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.xavier_uniform_(self.conv3.weight.data)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.xavier_uniform_(self.conv4.weight.data)
        nn.init.constant_(self.conv4.bias.data, 0)
        nn.init.xavier_uniform_(self.conv5.weight.data)
        nn.init.constant_(self.conv5.bias.data, 0)
        nn.init.xavier_uniform_(self.conv6.weight.data)
        nn.init.constant_(self.conv6.bias.data, 0)
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()