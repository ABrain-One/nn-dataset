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
        self.layer5 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0])
        )
        self._is_fwd_comp = False
        self.to(device)

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

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm)

    def build_layer(self, in_channels, out_channels, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def build_resblock(self, in_channels, out_channels):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
        self.optimizer = self.optimizer or self.build_optimizer(prm)

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
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def is_forward_compatible(self):
        return not self._is_fwd_comp

    def get_memory_breakdown(self):
        return {}

    def get_latency_breakdown(self):
        return {}

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def get_input_resolution(self):
        return (32, 32)

    def get_output_resolution(self):
        return (6, 6)

    def get_input_channels(self):
        return 3

    def get_output_channels(self):
        return 10

    def get_hyperparameters(self):
        return {'lr': 0.01, 'momentum': 0.9}

    def get_architecture(self):
        return 'MobileNetV2-like'

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer_config(self):
        return {'weight_decay': 5e-4}

    def get_dataset_config(self):
        return {'resolution': (32, 32), 'channels': 3}

    def get_model_config(self):
        return {'params_limit': 500000}

    def get_deployment_config(self):
        return {'input_size': (3, 32, 32), 'input_format': 'channels_first'}

    def get_training_config(self):
        return {'epochs': 10, 'batch_size': 32}

    def get_evaluation_config(self):
        return {'batch_size': 32}

    def get_training_data_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_evaluation_data_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def train_setup_with_data(self, prm, dataset):
        self.train_setup(prm)
        self.data_loader = self.get_training_data_loader(dataset, prm['batch_size'])

    def learn_with_data(self, dataset, prm):
        self.train_setup_with_data(prm, dataset)
        self.learn(self.data_loader)

    def get_next_hyperparameters(self, prm):
        return prm

    def get_best_hyperparameters(self, prm):
        return prm

    def get_next_architecture(self):
        return 'MobileNetV2-like'

    def get_best_architecture(self):
        return 'MobileNetV2-like'

    def get_next_optimizer_config(self, prm):
        return prm

    def get_best_optimizer_config(self, prm):
        return prm

    def get_next_dataset_config(self, prm):
        return prm

    def get_best_dataset_config(self, prm):
        return prm

    def get_next_model_config(self, prm):
        return prm

    def get_best_model_config(self, prm):
        return prm

    def get_next_deployment_config(self, prm):
        return prm

    def get_best_deployment_config(self, prm):
        return prm