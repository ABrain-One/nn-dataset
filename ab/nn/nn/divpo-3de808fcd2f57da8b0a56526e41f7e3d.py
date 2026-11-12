import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class SBlock(nn.Module):
    def forward(self, x):
        return x
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.prm = prm
        self.learnt_pad = self.in_channels % 16
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            SBlock(),
        )
        self.block_0 = SBlock()
        self.block_1 = SBlock()
        self.block_2 = SBlock()
        self.block_3 = SBlock()
        self.block_4 = SBlock()
        self.block_5 = SBlock()
        self.block_6 = SBlock()
        self.block_7 = SBlock()
        self.block_8 = SBlock()
        self.block_9 = SBlock()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(32, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.learnt_pad > 0:
            pad = torch.nn.ConstantPad2d(self.learnt_pad, 0)
            x = pad(x)
        x = self.stem(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.block_9(x)
        x = self.head(x)
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