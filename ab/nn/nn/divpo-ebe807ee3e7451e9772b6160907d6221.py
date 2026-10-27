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
        # Fused block for edge-friendly deployment
        if isinstance(self.conv[1], nn.BatchNorm2d):
            fused = (self.conv[0].weight, self.conv[0].bias,
                     self.conv[1].weight, self.conv[1].bias,
                     self.conv[2].weight)
            self.conv = torch.nn.utils.fuse_conv_bn_weights(
                *fused, kernel_size=3, stride=1, padding=1
            )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Follow the shape convention for CIFAR-10 (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Block(64, 64, stride=1)
        )
        self.layer1 = Block(64, 128, stride=2)
        self.layer2 = Block(128, 256, stride=2)
        self.layer3 = Block(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer3(x)
        x = self.avg_pool(x)
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, eval_data):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in eval_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def export(self):
        fused_block = Block(in_channels=64, out_channels=64, stride=1)
        fused_block.fuse()
        return fused_block