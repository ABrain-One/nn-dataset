import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0] # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to_parameters(prm, 500000)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = self.block6(x)
        x = self.avg_pool(x)
        x = self.flat(x)
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
            self.optimizer.step()

    def to_parameters(self, prm, total_limit):
        target_under = total_limit * 0.8
        # 3-layer block + 1-layer linear <= target_under
        target_blocks = ((target_under - 3) / 2 + 1) // 2
        # 17_11.3_1_12_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3_1_2_1.3