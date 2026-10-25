import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            Block(32, 64, stride=2),
            Block(64, 64),
            Block(64, 128, stride=2),
            Block(128, 128)
        )
        self.layer1 = nn.Sequential(Block(128, 128, groups=128), Block(128, 128))
        self.layer2 = nn.Sequential(
            Block(128, 256, stride=2),
            Block(256, 256)
        )
        self.layer3 = nn.Sequential(
            Block(256, 256, groups=256),
            Block(256, 256)
        )
        self.layer4 = nn.Sequential(
            Block(256, 512, stride=2),
            Block(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
            weight_decay=prm.get('weight_decay', 0.0001)
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

# Dummy input to derive the correct in_channels and out_shape
dummy_input = torch.randn(1, 3, 32, 32)
in_channels = dummy_input.shape[1]
out_shape = (10,)

# Parameters within the budget and the above constraints
prm = {
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'dropout': 0.2
}
device = torch.device('cpu')

# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Target number of parameters
target_params = 500000

# Tune the model to meet or slightly exceed the parameter limit
def tune_model_for_params(trial):
    net = Net(in_shape=(in_channels, 32, 32), out_shape=out_shape, prm=prm, device=device)
    prev_params = count_parameters(net)

    # Optimize the number of layers and their complexity
    for _ in range(2):  # Typically 2-3 layers are enough for the parameter limit
        blocks = []
        in_channels = 32

        for _ in range(trial.suggest_int('num_blocks', 1, 3)):
            out_channels = trial.suggest_int('out_channels', 32, 128, step=32)
            kernel_size = trial.suggest_int('kernel_size', 3, 5, step=2)
            stride = trial.suggest_int('stride', 1, 2)
            padding = (kernel_size - 1) // 2
            groups = trial.suggest_categorical('groups', [1, 2, 4, 8])

            blocks.append(Block(in_channels, out_channels, kernel_size, stride, padding, groups))
            in_channels = out_channels

        # Add a classifier with a fixed number of units to avoid overfitting
        classifier_units = 4096
        blocks.append(nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, classifier_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(classifier_units, num_classes)
        ))

        net.stem = nn.Sequential(*[nn.Sequential(*blocks[:2])])
        net.layer1 = nn.Sequential(*blocks[2:4])
        net.layer2 = nn.Sequential(*blocks[4:6])
        net.layer3 = nn.Sequential(*blocks[6:8])
        net.layer4 = nn.Sequential(*blocks[8:10])
        net.classifier = blocks[-1]

        current_params = count_parameters(net)
        if current_params > target_params:
            continue  # Try a different configuration

        net.to(device)
        net.optimizer = torch.optim.SGD(
            net.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
        )
        net.criteria = (nn.CrossEntropyLoss().to(device),)

        # Train for a single epoch to check the model's performance
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            net.train()
            for inputs, labels in train_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                net.optimizer.zero_grad()
                outputs = net(inputs)
                loss = net.criteria[0](outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_