import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.channels = [64, 128, 256]
        self.features = nn.Sequential(nn.Conv2d(in_shape[1], self.channels[0], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels[0]), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels[1]), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.channels[2]), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        final_channels = self.channels[2]
        classifier_input_dim = final_channels * 6 * 6
        self.classifier = nn.Sequential(nn.Dropout(p=prm.get('dropout', 0.5)), nn.Linear(classifier_input_dim, 4096), nn.ReLU(inplace=True), nn.Dropout(p=prm.get('dropout', 0.5)), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, out_shape[0]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

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