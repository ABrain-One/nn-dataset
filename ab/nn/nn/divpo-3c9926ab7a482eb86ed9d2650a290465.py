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
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = self._make_block(64, 128, 2, True)
        self.block3 = self._make_block(128, 256, 2, True)
        self.block4 = self._make_block(256, 512, 2, True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, out_shape[0])

    def _make_block(self, in_channels, out_channels, layers, batch_norm):
        layers_list = []
        for i in range(layers):
            layers_list += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

                                        
                                       

### Budget: 500000 parameters
                                       

### Edge-friendly latency (time budget) is tight. Optimize for minimal runtime after the first epoch.
                                       

### Structure and hyperparameter constraints follow above guidelines. Derive in_channels from in_shape. Do NOT hardcode the channel count.
                                       
### Class: Net
                                       

### Supported hyperparameters (list): 'lr', 'momentum'
                                       
### Fallback (if this cell fails): try running the last block of code independently, or use a different architecture template