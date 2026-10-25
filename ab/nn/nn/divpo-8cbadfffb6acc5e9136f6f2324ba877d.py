import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio):
        super(InvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, in_channels * expansion_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expansion_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(in_channels * expansion_ratio, in_channels * expansion_ratio, 3, stride, 1, groups=in_channels * expansion_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expansion_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(in_channels * expansion_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.channel_inverted_residuals = InvertedResidual(32, 32, 1, 3)
        self.channel_inverted_residuals_2 = InvertedResidual(32, 32, 1, 3)
        self.channel_inverted_residuals_3 = InvertedResidual(32, 32, 1, 3)

        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.channel_inverted_residuals(x)
        x = self.channel_inverted_residuals_2(x)
        x = self.channel_inverted_residuals_3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()