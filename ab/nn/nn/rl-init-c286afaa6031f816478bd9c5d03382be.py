import torch, torch.nn as nn

class GeGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_proj = self.proj(x)
        x, gate = x_proj.chunk(2, dim=1)
        return self.bn(x * torch.sigmoid(gate))

                            



def supported_hyperparameters():
    return {'lr': 0.01, 'momentum': 0.9}                                     

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = 3                                    
        self.image_size = in_shape[2]                                
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']                                   
        self.momentum = prm['momentum']                              

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers.append(GeGLU(self.in_channels, 128))                              
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        self._last_channels = 128

        layers.append(GeGLU(128, 256))                              
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(256))
        self._last_channels = 256

        layers.append(GeGLU(256, 512))                              
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(512))
        self._last_channels = 512

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()