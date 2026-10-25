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
        self.block1 = self._make_block(32, 64, 2, prm.get('dropout', 0.0))
        self.block2 = self._make_block(64, 128, 2, prm.get('dropout', 0.0))
        self.block3 = self._make_block(128, 256, 2, prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )
        self._set_fuse_coef(prm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)

    def _make_block(self, in_channels, out_channels, layers, dropout):
        layers_list = []
        for i in range(layers):
            layers_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        return nn.Sequential(*layers_list)

    def _set_fuse_coef(self, prm):
        self.fuse_coef = prm.get('fuse_coef', 1.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _get_name_postfix(self):
        return ''

    def _get_num_classes(self):
        return 10

    def _get_input_channel(self):
        return 3

    def _get_pretrained_path(self):
        return ''

    def _get_pretrained_layer_names(self):
        return []

    def _get_pretrained_weights(self):
        return {}

    def _get_train_layers(self):
        return []

    def _get_train_layer_names(self):
        return []

    def _get_train_parameters(self):
        return []

    def _get_train_parameters_with_weight_decay(self, weight_decay):
        return []

    def _get_train_parameters_with_momentum(self, momentum):
        return []

    def _get_train_parameters_with_lr(self, lr):
        return []

    def _get_train_parameters_with_lr_and_momentum(self, lr, momentum):
        return []

    def _get_train_parameters_with_lr_and_weight_decay(self, lr, weight_decay):
        return []

    def _get_train_parameters_with_lr_momentum_and_weight_decay(self, lr, momentum, weight_decay):
        return []

    def _get_train_parameters_without_lr_momentum_and_weight_decay(self):
        return []

    def _get_train_parameters_without_lr_and_momentum(self):
        return []

    def _get_train_parameters_without_lr_and_weight_decay(self):
        return []

    def _get_train_parameters_without_momentum_and_weight_decay(self):
        return []

    def _get_train_parameters_without_momentum(self):
        return []

    def _get_train_parameters_without_weight_decay(self):
        return []

    def _get_train_parameters_without_lr(self):
        return []

    def _get_train_parameters_without_lr_momentum(self):
        return []

    def _get_train_parameters_without_lr_weight_decay(self):
        return []

    def _get_train_parameters_without_lr_and_momentum_and_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_momentum(self):
        return []

    def _get_train_parameters_with_lr_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_and_momentum_and_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_and_momentum_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_momentum_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_and_weight_decay_momentum(self):
        return []

    def _get_train_parameters_with_lr_and_momentum(self):
        return []

    def _get_train_parameters_with_lr_and_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_weight_decay_momentum(self):
        return []

    def _get_train_parameters_with_lr_weight_decay_momentum(self):
        return []

    def _get_train_parameters_with_lr_and_momentum_momentum(self):
        return []

    def _get_train_parameters_with_lr_and_weight_decay_momentum(self):
        return []

    def _get_train_parameters_with_lr_momentum_momentum(self):
        return []

    def _get_train_parameters_with_lr_momentum_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_and_momentum_weight_decay(self):
        return []

    def _get_train_parameters_with_lr_and_momentum_momentum(self):
        return []