import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = nn.Sequential(
            self._make_layer(32, 3, 1),
            self._make_layer(64, 3, 2),
            self._make_layer(128, 3, 2)
        )
        self.classifier = nn.Linear(128, self.num_classes)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        layer = []
        layer += [nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layer += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(out_channels),
                     nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = torch.mean(x, dim=(2, 3))
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=1e-4
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
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=3.0)
            self.optimizer.step()