import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.dropout = prm['dropout']
        self.stem = nn.Sequential(nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature_extractor = nn.Sequential(self._make_dsu_block(64, 64, use_se=True), self._make_dsu_block(64, 128, use_se=True), self._make_dsu_block(128, 256, use_se=True), self._make_dsu_block(256, 512, use_se=True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, out_shape[0])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.Linear)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_dsu_block(self, in_channels, out_channels, use_se):
        layers = [nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, stride=1, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if use_se:
            reduction = 16
            fc_in = out_channels
            fc_out = fc_in // reduction
            layers.extend([nn.AdaptiveAvgPool2d(1), nn.Conv2d(fc_in, fc_out, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(fc_out, fc_in, kernel_size=1, stride=1, padding=0), nn.Sigmoid()])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9), weight_decay=0.0005)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)