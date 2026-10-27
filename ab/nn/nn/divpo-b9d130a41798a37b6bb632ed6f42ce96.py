import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_computed = False
        self.fwd_out_channels = 256
        self.fwd_out_shape = (256, 6, 6)

        self._size = len(in_shape)

        self._in_channels = in_shape[1] if self._size == 4 else in_shape[0]
        self._image_size = in_shape[2] if self._size == 4 else 32
        self._num_classes = out_shape[0]
        self._momentum = prm['momentum']
        self._learning_rate = prm['lr']

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum'],
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

    def _find_best_channel(self, channel, layer_id):
        if self._is_fwd_computed:
            return self.fwd_out_channels
        else:
            self._is_fwd_computed = True
            x = torch.zeros(1, channel, self._image_size,
                            self._image_size, device=self.device)
            x = self.features(x)
            self.fwd_out_channels = x[0].shape[0]
            self.fwd_out_shape = (self.fwd_out_channels, x[0].shape[2],
                                  x[0].shape[3])
            return self.fwd_out_channels

    def get_penultimate_channels(self, layer_id=0):
        if layer_id == 0:
            return self._in_channels
        else:
            return self._find_best_channel(self._in_channels, layer_id)

    def get_penultimate_layer_shape(self, layer_id=0):
        return (self.get_penultimate_channels(layer_id),
                self._image_size, self._image_size)

    def get_num_classes(self):
        return self._num_classes

    def get_classifier_input_shape(self):
        return self.get_penultimate_layer_shape()

    def get_classifier_input_channels(self):
        return self.get_penultimate_channels()