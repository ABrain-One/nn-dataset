import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (B, C, H, W)
        num_classes = int(out_shape[0])  # number of target classes
        self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.feature_extractor = self.build_feature_extractor()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def build_feature_extractor(self):
        layers = []
        inchans = 16
        for outchan in [16, 32, 64]:
            layers += self.make_block(inchans, outchan, 3)
            inchans = outchan
        return nn.Sequential(*layers)

    def make_block(self, inchans, outchan, kersz):
        layers = []
        layers += [
            nn.Conv2d(inchans, outchan, kernel_size=kersz, padding=1, bias=False),
            nn.BatchNorm2d(outchan),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchan, outchan, kernel_size=kersz, padding=1, bias=False),
            nn.BatchNorm2d(outchan),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            self.optimizer.step()