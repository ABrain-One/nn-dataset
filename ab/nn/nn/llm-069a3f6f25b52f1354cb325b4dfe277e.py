import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        self.entry = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = self.make_layer(64, 128, 2)
        self.layer2 = self.make_layer(128, 256, 2)
        self.layer3 = self.make_layer(256, 512, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, out_shape[0])

    def make_layer(self, in_ch, out_ch, n):
        blocks = []
        blocks += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)]
        blocks += [nn.BatchNorm2d(out_ch)]
        blocks += [nn.ReLU(inplace=True)]
        for _ in range(1, n):
            blocks += [nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)]
            blocks += [nn.BatchNorm2d(out_ch)]
            blocks += [nn.ReLU(inplace=True)]
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)