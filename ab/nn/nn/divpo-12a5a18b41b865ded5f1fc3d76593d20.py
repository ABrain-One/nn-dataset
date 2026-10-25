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
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fpn = None
        self.to(device)

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def build_anchor(self, in_shape):
        self.anchor_size = None
        self._is_fpn = False
        self.in_channels = in_shape[1]
        return None

    def add_level(self, level, level_params):
        if self._is_fpn:
            raise ValueError('Cannot add FPN levels to a model with existing FPN.')
        self._is_fpn = True
        in_channels = level_params['channels']
        num_anchors = level_params['num_anchors']
        anchor_size = level_params['anchor_size']
        num_classes = level_params['num_classes']
        if self.anchor_size is None:
            self.anchor_size = anchor_size
        elif anchor_size != self.anchor_size:
            raise ValueError(f'Anchor size mismatch: {anchor_size} != {self.anchor_size}')
        self.level_layers = nn.ModuleList([
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        ])
        self.level_pred_conf = nn.Conv2d(in_channels, num_anchors, kernel_size=3, padding=1)
        self.level_pred_cls = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.level_softmax = nn.Softmax(dim=1)
        self.level_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        if self._is_fpn:
            pred_conf = self.level_pred_conf(x)
            pred_cls = self.level_pred_cls(x)
            pred_cls = self.level_softmax(pred_cls)
            x = torch.cat((pred_conf, pred_cls), 1)
            x = self.level_layers(x)
        return self.classifier(x)