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
        # Fused weight and bias, to reduce params and have a fast path
        if self.conv.bias is None:
            fused_weight = self.conv.weight.clone()
        else:
            fused_weight = torch.empty_like(self.conv.weight)
            torch.addmm(fused_weight, self.conv.weight, self.batchnorm.weight.unsqueeze(-1), alpha=self.batchnorm.eps)
            torch.add(fused_weight, fused_weight, self.conv.bias.unsqueeze(-1), alpha=self.batchnorm.running_mean)
            torch.div(fused_weight, self.batchnorm.running_var.unsqueeze(-1).sqrt())
            torch.add(fused_weight, fused_weight, self.batchnorm.bias.unsqueeze(-1))

        return Block(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, fused_weight=fused_weight)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # Handle channel-first input shape
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(self.in_channels, 64, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(128, 256, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(256, 512, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(512, 512, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(512 * 2 * 2, num_classes)
        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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

    def supported_hyperparameters(self):
        return {'lr', 'momentum', 'weight_decay'}
    def export(self):
        return self.features, self.classifier

    def fuse(self):
        # Apply block-level fusion to reduce parameters
        fused_features = []
        for layer in self.features:
            if isinstance(layer, Block) and layer.fuse:
                fused_features.append(layer.fuse())
            else:
                fused_features.append(layer)
        self.features = nn.Sequential(*fused_features)