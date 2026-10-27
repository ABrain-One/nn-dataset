import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self._last_layer_name = None
        self._last_dim = None
        self.to(device)

      
    def build_layer(self, in_channels, out_channels, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _get_last_layer_name(self):
        return self._last_layer_name

    def _get_last_dim(self):
        return self._last_dim

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

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def _calc_out_channels(self, in_shape):
        self._last_layer_name = None
        self._last_dim = None
        with torch.no_grad():
            x = torch.empty(in_shape)
            x = self.stem(x)
            x = self.layer1(x)
            self._last_layer_name = 'layer1'
            self._last_dim = x.shape[1]
            x = self.layer2(x)
            self._last_layer_name = 'layer2'
            self._last_dim = x.shape[1]
            x = self.layer3(x)
            self._last_layer_name = 'layer3'
            self._last_dim = x.shape[1]
            x = self.layer4(x)
            self._last_layer_name = 'layer4'
            self._last_dim = x.shape[1]
        return self._last_dim

    def _calc_out_shape(self, in_shape):
        self._last_layer_name = None
        self._last_dim = None
        with torch.no_grad():
            x = torch.empty(in_shape)
            x = self.stem(x)
            x = self.layer1(x)
            self._last_layer_name = 'layer1'
            self._last_dim = x.shape[1]
            x = self.layer2(x)
            self._last_layer_name = 'layer2'
            self._last_dim = x.shape[1]
            x = self.layer3(x)
            self._last_layer_name = 'layer3'
            self._last_dim = x.shape[1]
            x = self.layer4(x)
            self._last_layer_name = 'layer4'
            self._last_dim = x.shape[1]
        out_channels = self._last_dim
        in_channels = in_shape[1]
        out_shape = (out_channels,)
        return out_shape

    def get_out_channels(self, in_shape):
        self._last_layer_name = None
        self._last_dim = None
        with torch.no_grad():
            x = torch.empty(in_shape)
            x = self.stem(x)
            x = self.layer1(x)
            self._last_layer_name = 'layer1'
            self._last_dim = x.shape[1]
            x = self.layer2(x)
            self._last_layer_name = 'layer2'
            self._last_dim = x.shape[1]
            x = self.layer3(x)
            self._last_layer_name = 'layer3'
            self._last_dim = x.shape[1]
            x = self.layer4(x)
            self._last_layer_name = 'layer4'
            self._last_dim = x.shape[1]
        return self._last_dim

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

    def _get_last_layer_name(self):
        return self._last_layer_name

    def _get_last_dim(self):
        return self._last_dim

    def _get_layer_channel(self, layer_name):
        if layer_name == 'layer1':
            return self._last_dim
        if layer_name == 'layer2':
            return self._last_dim
        if layer_name == 'layer3':
            return self._last_dim
        if layer_name == 'layer4':
            return self._last_dim
        raise ValueError('Invalid layer name')

    def _get_layer_index(self, layer_name):
        if layer_name == 'layer1':
            return 0
        if layer_name == 'layer2':
            return 1
        if layer_name == 'layer3':
            return 2
        if layer_name == 'layer4':
            return 3
        raise ValueError('Invalid layer name')

    def _get_layer_channels(self, layer_name):
        layer_index = self._get_layer_index(layer_name)
        return self._get_layer_channel(layer_name)

    def _get_layer_channels_and_out_shape(self, layer_name):
        layer_index = self._get_layer_index(layer_name)
        layer_channels = self._get_layer_channel(layer_name)
        out_channels = layer_channels
        out_shape = (out_channels,)
        return layer_channels, out_shape

    def _get_layer_channels_and_out_channels(self, layer_name):
        layer_index = self._get_layer_index(layer_name)
        layer_channels = self