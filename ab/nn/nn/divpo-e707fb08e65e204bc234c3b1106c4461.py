import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.in_size = (32, 32)
        self.dropout = prm.get('dropout', 0.2)
        self.block = Block(self.in_channels, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers = nn.Sequential(
            Block(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.classifier = nn.Linear(256, self.num_classes)
        self.apply_to_classifier_weights(self.block)

    def apply_to_classifier_weights(self, layer):
        if self.classifier is not None:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    self.classifier.weight.data = torch.nn.functional.normalize(self.classifier.weight.data)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        x = self.dropout_layer(x)
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