import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1),
            InvertedResidual(32, 32, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(64, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for k in self.features.state_dict():
            if k.endswith('conv.conv.weight'):
                s = self.features.state_dict()[k].shape
                self.features.state_dict()[k.replace('conv.conv.weight', '0.weight')] = self.features.state_dict()[k][:, :s[1]//6, :, :]
                self.features.state_dict()[k.replace('conv.conv.weight', '1.weight')] = self.features.state_dict()[k][:, s[1]//6:(s[1]//6)*2, :, :]
                self.features.state_dict()[k.replace('conv.conv.weight', '2.weight')] = self.features.state_dict()[k][:, (s[1]//6)*2:(s[1]//6)*3, :, :]
                self.features.state_dict()[k.replace('conv.conv.weight', '3.weight')] = self.features.state_dict()[k][:, (s[1]//6)*3:(s[1]//6)*4, :, :]
                self.features.state_dict()[k.replace('conv.conv.weight', '4.weight')] = self.features.state_dict()[k][:, (s[1]//6)*4:(s[1]//6)*5, :, :]
                self.features.state_dict()[k.replace('conv.conv.weight', '5.weight')] = self.features.state_dict()[k][:, (s[1]//6)*5:, :, :]

        self._last_channels = 128
        for k in prm:
            if k == 'lr': continue
            if k == 'momentum': continue
            if k == 'weight_decay': continue
            if k == 'dropout': continue

        self.classifier = nn.Linear(128, out_shape[0])
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
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