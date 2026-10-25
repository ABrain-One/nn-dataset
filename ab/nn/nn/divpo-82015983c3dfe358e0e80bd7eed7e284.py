import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = self._make_block(32, 64, 2, prm.get('dropout', 0.0))
        self.block2 = self._make_block(64, 128, 2, prm.get('dropout', 0.0))
        self.block3 = self._make_block(128, 256, 2, prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )
        self._size_record = [32 * 8 * 8]
        self._size_record_block1 = [32 * 4 * 4]
        self._size_record_block2 = [64 * 2 * 2]
        self._size_record_block3 = [128 * 2 * 2]
        self._size_record_avgpool = [256 * 6 * 6]
        self.to(device)

    def _make_block(self, in_channels, out_channels, layers, dropout):
        layers_list = []
        for i in range(layers):
            layers_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
            self._size_record_block1 = [out_channels * 8 * 8]
            self._size_record_block2 = [out_channels * 4 * 4]
            self._size_record_block3 = [out_channels * 2 * 2]
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )
        self.criteria = (nn.CrossEntropyLoss(),)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria[0](output, target)
            loss.backward()
            self.optimizer.step()