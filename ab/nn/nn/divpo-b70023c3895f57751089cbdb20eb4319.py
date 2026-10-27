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
        self.layer1 = self.build_layer(32, 32, 2, stride=1)
        self.layer2 = self.build_layer(32, 64, 2, stride=2)
        self.layer3 = self.build_layer(64, 128, 2, stride=2)
        self.layer4 = self.build_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False
        self._last_inner = None
        self.to(device)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.01)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm)

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for _ in range(blocks - 1):
            layers += [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            ]
        return nn.Sequential(*layers)

    def build_optimizer(self, prm):
        return torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 5e-4)
        )

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = self.build_optimizer(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def is_forward_compatible(self):
        return not self._is_fwd_comp

    def get_last_inner(self):
        return self._last_inner

    def get_inner(self, x, prm, is_fwd):
        if is_fwd and not self.is_forward_compatible():
            raise RuntimeError("Model is not forward compatible.")
        self._is_fwd_comp = is_fwd
        if self._last_inner is None:
            self._last_inner = nn.Identity()
        return self._last_inner

    def get_out(self, x):
        return x

    def get_out_act(self, x):
        return x

    def get_out_conv(self, x):
        return x

    def get_out_pool(self, x):
        return x

    def get_out_linear(self, x):
        return x

    def get_out_bn(self, x):
        return x

    def get_out_flatten(self, x):
        return x

    def get_out_avgpool(self, x):
        return x

    def get_out_maxpool(self, x):
        return x

    def get_out_dropout(self, x):
        return x

    def get_out_identity(self, x):
        return x

    def get_out_relu(self, x):
        return x

    def get_out_adaptiveavgpool(self, x):
        return x

    def get_out_adaptivemaxpool(self, x):
        return x

    def get_out_maxunpool2d(self, x):
        return x

    def get_out_instancenorm2d(self, x):
        return x

    def get_out_groupnorm(self, x):
        return x

    def get_out_batchnorm2d(self, x):
        return x

    def get_out_batchnorm1d(self, x):
        return x

    def get_out_batchnorm3d(self, x):
        return x

    def get_out_fractional_max_pool2d(self, x):
        return x

    def get_out_fractional_max_pool3d(self, x):
        return x

    def get_out_max_pool3d(self, x):
        return x

    def get_out_max_pool2d(self, x):
        return x

    def get_out_fractional_avg_pool2d(self, x):
        return x

    def get_out_fractional_avg_pool3d(self, x):
        return x

    def get_out_avg_pool3d(self, x):
        return x

    def get_out_avg_pool2d(self, x):
        return x

    def get_out_fractional_max_pool3d_backward(self, x):
        return x

    def get_out_fractional_max_pool2d_backward(self, x):
        return x

    def get_out_max_pool3d_backward(self, x):
        return x

    def get_out_max_pool2d_backward(self, x):
        return x

    def get_out_fractional_avg_pool3d_backward(self, x):
        return x

    def get_out_fractional_avg_pool2d_backward(self, x):
        return x

    def get_out_avg_pool3d_backward(self, x):
        return x

    def get_out_avg_pool2d_backward(self, x):
        return x

    def get_out_fractional_max_pool3d_update_indices(self, x):
        return x

    def get_out_fractional_max_pool2d_update_indices(self, x):
        return x

    def get_out_max_pool3d_update_indices(self, x):
        return x

    def get_out_max_pool2d_update_indices(self, x):
        return x

    def get_out_fractional_avg_pool3d_update_indices(self, x):
        return