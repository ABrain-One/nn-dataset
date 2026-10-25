import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly deployment
        if isinstance(self.conv, nn.Conv2d) and isinstance(self.batchnorm, nn.BatchNorm2d):
            fused_conv = nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True
            ).requires_grad_(False)

            fused_conv.weight.copy_(self.conv.weight)
            if self.conv.bias is not None:
                fused_conv.weight.add_(self.conv.bias)
            fused_conv.weight.copy_(self.batchnorm.weight.view(1, -1, 1, 1))

            # If the model is in eval mode, BatchNorm statistics are not updated.
            # This means that the running_mean and running_var in BatchNorm are equivalent to the mean and var in BN.
            # Thus, we can directly use them to compute the scale and bias.
            if not self.training:
                fused_conv.weight.div_(self.batchnorm.running_var.sqrt().view(1, -1, 1, 1).clamp_(min=1e-5))
                fused_conv.bias.copy_(-self.batchnorm.running_mean)
            self.fused_conv = fused_conv
            return self.fused_conv
        return self

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Handle the channels-first input shape (C, H, W)
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256, stride=1),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self._initialize_weights()

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
        )

    def forward(self, x):
        x = self.features(x)
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
            weight_decay=prm.get('weight_decay', 0.0001)
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

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.constant_(self.classifier.weight, 0)
        nn.init.constant_(self.classifier.bias, 0)

    def fuse(self):
        # Apply fused blocks for edge-friendly deployment
        self.features = torch.nn.Sequential(*map(lambda x: x.fuse() if hasattr(x, 'fuse') else x, self.features))
        self.to(self.device)