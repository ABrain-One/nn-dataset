import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly deployment
        if isinstance(self.relu, nn.Sequential):
            fused = [self.conv, self.relu[0], self.bn]
        else:
            fused = [self.conv, self.relu, self.bn]
        return nn.Sequential(*fused)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, stride=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

        self._compile(prm)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _compile(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def train_setup(self, prm):
        self._compile(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def get_loss(self):
        return self.criteria[0]

    def fuse(self):
        # Fuse blocks for edge-friendly deployment
        fused_features = []
        for block in self.features:
            if isinstance(block, Block):
                fused_features.append(block.fuse())
            else:
                fused_features.append(block)
        self.features = nn.Sequential(*fused_features)
        for block in list(self.classifier)[:-1]:
            if isinstance(block, nn.Linear):
                in_features = block.in_features
                out_features = list(self.classifier)[-1].out_features
                self.classifier = nn.Sequential(
                    nn.Dropout(p=prm.get('dropout', 0.2)),
                    nn.Linear(in_features, out_features),
                    list(self.classifier)[-1]
                )
                break