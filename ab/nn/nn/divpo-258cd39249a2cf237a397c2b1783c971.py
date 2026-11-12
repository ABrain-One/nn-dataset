import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.features = 32
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.conv1 = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=1)
        self.conv2 = nn.Conv2d(self.features, self.features, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU(num_parameters=1)
        self.conv3 = nn.Conv2d(self.features, self.features * 2, kernel_size=3, stride=2, padding=1)
        self.prelu3 = nn.PReLU(num_parameters=1)
        self.conv4 = nn.Conv2d(self.features * 2, self.features * 2, kernel_size=3, padding=1)
        self.prelu4 = nn.PReLU(num_parameters=1)
        self.conv5 = nn.Conv2d(self.features * 2, self.features * 2, kernel_size=3, padding=1)
        self.prelu5 = nn.PReLU(num_parameters=1)
        self.conv6 = nn.Conv2d(self.features * 2, self.features * 3, kernel_size=3, stride=2, padding=1)
        self.prelu6 = nn.PReLU(num_parameters=1)
        self.conv7 = nn.Conv2d(self.features * 3, self.features * 3, kernel_size=3, padding=1)
        self.prelu7 = nn.PReLU(num_parameters=1)
        self.conv8 = nn.Conv2d(self.features * 3, self.features * 3, kernel_size=3, padding=1)
        self.prelu8 = nn.PReLU(num_parameters=1)
        self.adapter = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(self.features * 3, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.conv5(x)
        x = self.prelu5(x)
        x = self.conv6(x)
        x = self.prelu6(x)
        x = self.conv7(x)
        x = self.prelu7(x)
        x = self.conv8(x)
        x = self.prelu8(x)
        x = self.adapter(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.device = next(self.parameters()).device
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

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