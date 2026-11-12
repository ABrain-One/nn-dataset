import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1, bias=False)
        self.bn_first_conv = nn.BatchNorm2d(self.features)
        self.ac_first_conv = nn.ReLU(inplace=True)
        self.second_group = self._make_layer(32, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.classifier = nn.Linear(32, self.num_classes)

    def _make_layer(self, output_features: int, blocks: int, stride: int = 1):
        block_ops = []
        if stride != 1:
            block_ops.append(nn.Conv2d(self.features, output_features, kernel_size=3, padding=1, stride=stride, bias=False))
            block_ops.append(nn.BatchNorm2d(output_features))
        for i in range(blocks):
            block_ops.append(
                nn.Conv2d(self.features, output_features, kernel_size=3, padding=1, stride=1, bias=False))
            block_ops.append(nn.BatchNorm2d(output_features))
            block_ops.append(nn.ReLU(inplace=True))
        self.features = output_features
        return nn.Sequential(*block_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.bn_first_conv(x)
        x = self.ac_first_conv(x)
        x = self.second_group(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=1e-4,
            nesterov=True
        )
        
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()