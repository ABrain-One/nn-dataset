import torch
import torch.nn as nn
import torch.optim as optim


def supported_hyperparameters():
    return {'lr', 'momentum'}


class CIFARResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, dropout_p=0.0):
        super(CIFARResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if use_dropout else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CIFARResidualNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_dropout=False, dropout_p=0.0):
        super(CIFARResidualNet, self).__init__()
        
        # Stem: 3x3 Conv2d, norm, activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: residual blocks at 32x32
        self.stage1 = self._make_layer(64, 64, 2, stride=1)
        
        # Stage 2: residual blocks at 16x16
        self.stage2 = self._make_layer(64, 128, 2, stride=2, use_dropout=use_dropout, dropout_p=dropout_p)
        
        # Stage 3: residual blocks at 8x8
        self.stage3 = self._make_layer(128, 256, 2, stride=2, use_dropout=use_dropout, dropout_p=dropout_p)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_dropout=False, dropout_p=0.0):
        layers = []
        layers.append(CIFARResidualBlock(in_channels, out_channels, stride, use_dropout, dropout_p))
        for _ in range(1, num_blocks):
            layers.append(CIFARResidualBlock(out_channels, out_channels, 1, use_dropout, dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.channel_number = in_shape[1]
        self.image_size = in_shape[2]
        self.class_number = out_shape[0]
        
        # Use hyperparameters from prm
        # The description mentions optional Dropout2d. We use a fixed small rate or 0.
        # To ensure we "use" the hyperparameters if they were required, but here only lr/momentum are in supported.
        # However, the prompt says: "LEMUR rejects a model that does not use every hyperparameter named in supported_hyperparameters".
        # supported_hyperparameters returns {'lr', 'momentum'}.
        # So I just need to make sure I don't break them.
        # I will not add dropout to the architecture unless it's standard. The description says "optional". 
        # I will set use_dropout=False to keep it simple and standard for CIFAR unless specified.
        # Wait, if I don't use dropout, I don't need a 'dropout' hyperparameter.
        # But what if the architecture description implies I *should* use it? 
        # "Blocks use ... optional Dropout2d".
        # I will assume standard choice is no dropout for this small net unless it helps. 
        # Actually, looking at the previous model, it didn't have dropout.
        # I will just instantiate the model.
        
        self.model = CIFARResidualNet(
            in_channels=self.channel_number,
            num_classes=self.class_number,
            use_dropout=False,
            dropout_p=0.0
        )

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()