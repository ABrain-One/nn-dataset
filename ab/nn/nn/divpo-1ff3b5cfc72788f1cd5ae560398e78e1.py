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
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_computed = False
        self.fwd_out_channels = 256

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self._is_fwd_computed = False

    
    def build_layer(self, in_channels, out_channels, layers, kernel_size=3, padding=1, dropout=0.0):
        layer_list = []
        for i in range(layers):
            layer_list.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            )
            layer_list.append(nn.BatchNorm2d(out_channels))
            layer_list.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            if dropout > 0:
                layer_list.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layer_list)

    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

    def compute_fwd_out_channels(self, x):
        if not self._is_fwd_computed:
            self.fwd_out_channels = self(x).shape[1]
            self._is_fwd_computed = True
        return self.fwd_out_channels