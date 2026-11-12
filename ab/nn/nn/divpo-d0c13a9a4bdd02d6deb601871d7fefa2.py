import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.drop1 = nn.Dropout(prm.get('dropout', 0.25))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.drop2 = nn.Dropout(prm.get('dropout', 0.25))

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv41 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.drop3 = nn.Dropout(prm.get('dropout', 0.25))

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv51 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.drop4 = nn.Dropout(prm.get('dropout', 0.25))

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(256 * 7 * 7, 10)

        self.channels = 32
        self.features = 32
        self.conv_count = 0
        self.drop_count = 0
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.optimizer = None
        self.criteria = None
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.drop1(self.conv21(x))

        x = self.conv3(x)
        x = self.drop2(self.conv31(x))

        x = self.conv4(x)
        x = self.drop3(self.conv41(x))

        x = self.conv5(x)
        x = self.drop4(self.conv51(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()