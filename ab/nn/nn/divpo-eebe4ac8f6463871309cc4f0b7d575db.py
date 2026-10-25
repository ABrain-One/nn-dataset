import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# This is the pre-defined configuration for the Net class.
CFG = {
    'name': 'Net',
    'model_type': 'PyTorch_Classifier',
    'in_channels': 3,
    'num_classes': 10,
    'channels': [32, 64, 128, 256],
    'kernel_sizes': [3, 3, 3, 3],
    'strides': [1, 2, 2, 2],
    'paddings': [1, 1, 1, 1],
    'drop_probs': [0.0, 0.0, 0.0, 0.0],
    'hyper_parameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'epochs': 1
    },
    'mode': 'training',
    'device': 'cpu',
    'dropout': 0.2
}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.weight_decay = prm.get('weight_decay', 0.0005)

        layers = []
        in_channels = self.in_channels
        for i, (out_channels, kernel_size, stride, padding, drop_prob) in enumerate(zip(CFG['channels'], CFG['kernel_sizes'], CFG['strides'], CFG['paddings'], CFG['drop_probs'])):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(drop_prob))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(in_channels * 6 * 6, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', self.learning_rate),
            momentum=prm.get('momentum', self.momentum),
            weight_decay=prm.get('weight_decay', self.weight_decay)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()