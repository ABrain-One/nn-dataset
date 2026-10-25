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
        self.layer4 = self.build_resblock(128, 128)
        self.layer5 = self.build_layer(128, 128, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self._last_layer_name = 'classifier'
        self._last_layer_multiplier = 1
        self._last_layer_channels = 128
        self._last_layer_kernel_size = 1
        self._last_layer_stride = 1
        self._last_layer_padding = 0
        self._last_layer_dilation = 1
        self._last_layer_groups = 1
        self._last_layer_bias = True
        self._last_layer_batch_norm = False
        self._last_layer_dropout = 0.0
        self._last_layer_in_channels = 128
        self._last_layer_out_channels = out_shape[0]
        self._last_layer_input_height = 6
        self._last_layer_input_width = 6
        self._last_layer_output_height = 6
        self._last_layer_output_width = 6
        self._last_layer_stride_h = 1
        self._last_layer_stride_w = 1

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

        self.to(device)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.configure_optimizers(prm)

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(self.build_block(in_channels, out_channels, stride))
            else:
                layers.append(self.build_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def build_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def build_resblock(self, in_channels, out_channels):
        return nn.Sequential(
            self.build_block(in_channels, out_channels),
            self.build_block(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def configure_optimizers(self, prm):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )