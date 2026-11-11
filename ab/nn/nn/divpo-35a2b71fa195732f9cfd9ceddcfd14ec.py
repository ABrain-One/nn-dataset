import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        short = self.downsample(x)
        fea = self.conv1(x)
        fea = self.bn1(fea)
        fea = self.relu(fea)
        fea = self.conv2(fea)
        fea = self.bn2(fea)
        fea += short
        fea = self.relu(fea)
        return fea

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.small_block1 = SmallBlock(32, 32)
        self.small_block2 = SmallBlock(32, 64, stride=2)
        self.small_block3 = SmallBlock(64, 128, stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feas = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.small_block1(x)
        x = self.small_block2(x)
        x = self.small_block3(x)
        x = self.average_pool(x)
        x = x.view(x.size(0), -1)
        x = self.feas(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm['lr'], momentum=prm['momentum'],
            weight_decay=1e-4, dampening=0, nesterov=False
        )

    def learn(self, train_data):
        self.train()
        for data, target in train_data:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            self.optimizer.step()