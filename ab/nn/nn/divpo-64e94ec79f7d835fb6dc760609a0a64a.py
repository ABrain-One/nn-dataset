import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle in_shape as (batch, channels, height, width)
        num_classes = out_shape[0]

        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=1),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, num_classes)
        )

        self._initialize_weights()

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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum', 'weight_decay'}