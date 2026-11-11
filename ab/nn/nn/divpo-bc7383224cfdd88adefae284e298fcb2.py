import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SmallBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        short = self.downsample(x)
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.conv2(feat)
        feat = self.bn2(feat)
        feat += short
        feat = self.relu(feat)
        return feat

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.small_block1 = SmallBlock(32, 32)
        self.small_block2 = SmallBlock(32, 64, 2)
        self.small_block3 = SmallBlock(64, 128, 2)
        self.average_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.features = nn.Linear(128 * 7 * 7, 128)
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.first_conv(x), inplace=True)
        x = self.small_block1(x)
        x = self.small_block2(x)
        x = self.small_block3(x)
        x = self.average_pool(x)
        x = x.flatten(1)
        x = nn.functional.relu(self.features(x))
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=prm['lr'], momentum=prm['momentum'],
            weight_decay=1e-4
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self(inputs)
            loss = self.criteria(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()