import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def fuse(self):
        # Fuse Conv and BatchNorm for optimization
        if isinstance(self.conv[1], nn.BatchNorm2d):
            fused_conv = (self.conv[0], self.conv[1], self.conv[2])
            self.__class__ = nn.Conv2d
            self.in_channels = self.conv[0].in_channels
            self.out_channels = self.conv[0].out_channels
            self.kernel_size = self.conv[0].kernel_size
            self.groups = self.conv[0].groups
            self.stride = self.conv[0].stride
            self.padding = self.conv[0].padding
            self.dilation = self.conv[0].dilation
            self.weight = fused_conv[0].weight
            self.bias = fused_conv[0].bias
            self.running_mean = fused_conv[1].running_mean
            self.running_var = fused_conv[1].running_var
            self.num_batches_tracked = fused_conv[1].num_batches_tracked
            self.eps = fused_conv[1].eps
            self.affine = fused_conv[1].affine
            del self.conv
            self.conv = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # derive from in_shape
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(self.in_channels, 64, stride=1),
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

        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
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