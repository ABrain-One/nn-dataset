import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
            
        if dropout_rate > 0:
            self.dropout_after_shortcut = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        
        out += self.shortcut(x)
        
        if hasattr(self, 'dropout_after_shortcut'):
            out = self.dropout_after_shortcut(out)
        
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])

        self.stem = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1, dropout_rate=prm.get('dropout', 0.0)),
            BasicBlock(64, 64, dropout_rate=prm.get('dropout', 0.0))
        )
        
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, dropout_rate=prm.get('dropout', 0.0)),
            BasicBlock(128, 128, dropout_rate=prm.get('dropout', 0.0))
        )
        
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, dropout_rate=prm.get('dropout', 0.0)),
            BasicBlock(256, 256, dropout_rate=prm.get('dropout', 0.0))
        )
        
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, dropout_rate=prm.get('dropout', 0.0)),
            BasicBlock(512, 512, dropout_rate=prm.get('dropout', 0.0))
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.dropout_classifier = nn.Dropout(prm.get('dropout', 0.0)) if 'dropout' in prm else None

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout_classifier:
            x = self.dropout_classifier(x)
        
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0)
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
            self.optimizer.step()