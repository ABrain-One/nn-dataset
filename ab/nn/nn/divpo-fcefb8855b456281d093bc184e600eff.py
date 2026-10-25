import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fuse batch norm and ReLU for faster inference
        if self.conv.in_channels == self.bn.running_mean.shape[0]:
            fused_conv = nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                bias=True
            ).requires_grad_(False)

            # prepare mask
            mask = torch.zeros(self.conv.out_channels,
                               self.conv.in_channels,
                               self.conv.kernel_size[0],
                               self.conv.kernel_size[1])

            mask[:, :, (self.conv.kernel_size[0] - 1) // 2, (self.conv.kernel_size[1] - 1) // 2] = 1

            weight = self.conv.weight * mask
            bias = (self.conv.weight.sum(dim=[2, 3]) - self.conv.weight[:, :, (self.conv.kernel_size[0] - 1) // 2,
                                           (self.conv.kernel_size[1] - 1) // 2] * self.conv.padding).view(-1)
            fused_conv.weight.data = weight
            fused_conv.bias.data = torch.tensor(self.bn.running_mean) * self.bn.weight + bias
            return fused_conv
        else:
            return self

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape for CIFAR-10 (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            Block(32, 64, kernel_size=3, stride=2, padding=1),
            Block(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = Block(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Sequential(
            Block(128, 256, kernel_size=3, stride=2, padding=1),
            Block(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            Block(256, 512, kernel_size=3, stride=2, padding=1),
            Block(512, 512, kernel_size=3, stride=1, padding=1),
            Block(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)
        self._initialize_weights()

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.weight, 0)
        nn.init.constant_(self.classifier.bias, 0)

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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def load_weights(self, weights):
        self.load_state_dict(weights, strict=False)