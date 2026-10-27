import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # Follow the guide for this variable
        num_classes = out_shape[0]
        self.conv1 = Block(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = Block(128, 256, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
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
            momentum=prm.get('momentum', 0.9)
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

    def __next_block_name(self, idx):
        return f'conv{idx+1}'

    def __next_name(self, idx):
        block = self.__next_block_name(idx)
        postfix = ''
        if idx == 0:
            postfix = 'conv1'
        elif idx == 1:
            postfix = 'conv2'
        elif idx == 2:
            postfix = 'conv3'
        return f'{postfix}.{block}.{["conv", "bn", "relu"][idx % 3]}'

    def __getitem__(self, idx):
        if idx < 0 or idx >= 3:
            raise IndexError("index out of range")
        name = self.__next_name(idx)
        if 'conv' in name:
            return getattr(self, name)
        elif 'bn' in name:
            return getattr(self, name).bn
        elif 'relu' in name:
            return getattr(self, name).relu
        raise ValueError("invalid index")

    def __len__(self):
        return 3

def learn(model, train_data, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_setup(prm)
    model.to(device)
    optimizer = model.optimizer
    criteria = model.criteria
    for inputs, labels in train_data:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criteria[0](outputs, labels)
        loss.backward()
        optimizer.step()

def train_setup(model, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_setup(prm)
    model.to(device)
    optimizer = model.optimizer
    model.to(device)
    return model, optimizer, model.criteria

def get_parameter_groups(model, weight_decay=1e-5):
    parameter_groups = {}
    parameters = list(model.parameters())
    for i, param in enumerate(parameters):
        if param.requires_grad:
            if 'bn' in str(param.data_ptr()):
                parameter_groups['bn'] = {
                    'params': parameters[i+1:],
                    'weight_decay': weight_decay
                }
                break
            parameter_groups['others'] = {
                'params': parameters[:i+1],
                'weight_decay': weight_decay
            }
    return parameter_groups

def supported_hyperparameters():
    return {'lr', 'momentum'}