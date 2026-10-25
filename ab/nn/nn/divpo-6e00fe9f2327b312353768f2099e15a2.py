import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self.vgg_block(32, 64, 1, 1)
        self.layer2 = self.vgg_block(64, 128, 2, 2)
        self.layer3 = self.vgg_block(128, 256, 2, 2)
        self.layer4 = self.vgg_block(256, 512, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, out_shape[0])

    def vgg_block(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(blocks):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
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
            weight_decay=prm.get('weight_decay', 5e-4)
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_flops(self, input_size):
        # This is a rough estimate and might not be 100% accurate.
        # It's based on the assumption that conv layers are the most compute-intensive.
        total_flops = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Assuming input_size[1] is always 3 for RGB images
                total_flops += (input_size[1] * module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels)
        return total_flops

    def get_memory_usage(self, input_size):
        # Estimate memory based on parameters and input size
        # This is a rough estimate and might not be 100% accurate.
        mem = self.count_parameters() * 4  # Parameters are stored as FP32
        mem += input_size[0] * input_size[1] * input_size[2] * input_size[3] * 4  # Input data is also FP32
        return mem