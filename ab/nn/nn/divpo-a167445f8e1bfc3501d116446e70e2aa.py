import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        hidden_dim = int(input_channels * expand_ratio)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, output_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.act2 = nn.ReLU(inplace=False)

    def forward(self, x):
        if not self.use_res_connect:
            x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn3(self.conv3(self.bn2(self.conv2(x)))))
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 1e-2)
        self.momentum = prm.get('momentum', 0.9)
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 32, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(32, 64, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InvertedResidual(64, 64, 1, 1),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(64 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, data_bundle):
        self.train()
        for batch in data_bundle:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()