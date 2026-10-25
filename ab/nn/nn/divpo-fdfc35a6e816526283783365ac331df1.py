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
        if self.conv.bias is not None:
            fused_weight = self.conv.weight + torch.diag(self.bn.weight)
            bias = torch.nn.functional.batch_norm(
                self.conv.bias.view(1, -1, 1, 1),
                torch.zeros(1, self.conv.out_channels, 1, 1, dtype=self.conv.bias.dtype, device=self.conv.bias.device),
                self.bn.running_var.view(1, -1, 1, 1),
                self.bn.running_mean,
                self.bn.weight,
                True,
                self.bn.eps
            ).view(-1)
        else:
            fused_weight = torch.diag(self.bn.weight) @ self.conv.weight
            bias = torch.zeros_like(self.conv.bias)

        self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels,
                              kernel_size=self.conv.kernel_size,
                              stride=self.conv.stride,
                              padding=self.conv.padding,
                              dilation=self.conv.dilation,
                              groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias.data.copy_(bias)
        self.__delattr__('bn')

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # adhere to the overridden input channels rule
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(in_channels, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(256, 512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(512, 512, stride=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (6, 6))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()