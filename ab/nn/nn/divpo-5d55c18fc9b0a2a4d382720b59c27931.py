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
        self.layer1 = self.build_layer(32, 32, 2, dropout=prm.get('dropout', 0.0))
        self.layer2 = self.build_layer(32, 64, 2, dropout=prm.get('dropout', 0.0))
        self.layer3 = self.build_layer(64, 128, 2, dropout=prm.get('dropout', 0.0))
        self.layer4 = self.build_layer(128, 256, 2, dropout=prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.0)),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_computed = False
        self.fwd_out_shape = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self._fwd_out_shape = None

    def build_layer(self, in_channels, out_channels, layers, dropout):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(layers - 1)
        ] + [
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

    def _build_fwd_cases(self, x_in):
        if self._is_fwd_computed:
            return
        self._fwd_out_shape = []
        x = x_in
        for layer in self.children():
            x = layer(x)
            self._fwd_out_shape.append(x.shape[1])
        self._fwd_out_shape = sum(self._fwd_out_shape)
        self._is_fwd_computed = True

    def get_out_channels(self, in_channels):
        if not self._is_fwd_computed:
            self._build_fwd_cases(torch.empty(1, in_channels, 32, 32))
        return self._fwd_out_shape

    def get_flops(self, in_channels):
        flops = 0
        x = torch.empty(1, in_channels, 32, 32)
        for layer in self.children():
            x = layer(x)
            batch_size, C, H, W = x.size()
            flops += C * H * W
        return flops

    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)