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
        if isinstance(self.conv[1], nn.BatchNorm2d):
            fused_conv = torch.nn.quantized.FloatFunctional()
            q_conv = torch.quantization.quantize_dynamic(
                self.conv[0], {torch.nn.Conv2d}, dtype=torch.qint8
            )
            q_bn = torch.quantization.quantize_dynamic(
                self.conv[1], {torch.nn.BatchNorm2d}, dtype=torch.qint32
            )
            self.conv = nn.Sequential(
                fused_conv.conv2d_quantized(q_conv, q_bn, self.conv[2]),
            )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from input shape
        num_classes = int(out_shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 64, stride=2),
            Block(64, 128, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = Block(128, 128)
        self.layer2 = Block(128, 256, stride=2)
        self.layer3 = Block(256, 256)
        self.layer4 = Block(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
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