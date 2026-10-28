import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        _, c_in, h_in, w_in = in_shape
        if len(out_shape) == 1:
            n_classes = out_shape[0]
        else:
            n_classes = out_shape[1]
        start_chan = 64
        self.entry = nn.Sequential(nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=4, stride=4))
        layers = []
        in_channels = start_chan
        pool = 2
        layer_idx = 0
        while h_in // pool > 4 and w_in // pool > 4:
            out_channels = min(512, int(in_channels * 2 ** 0.3333333333333333))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            pool *= 2
            layer_idx += 1
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        classifier_input_size = in_channels * 4 * 4
        dropout = nn.Dropout(p=prm['dropout'])
        classifier = nn.Linear(classifier_input_size, n_classes)
        self.classifier = nn.Sequential(dropout, classifier)
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

    def forward(self, x):
        x = self.entry(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        criterion, = self.criteria
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        return (0.0, None)