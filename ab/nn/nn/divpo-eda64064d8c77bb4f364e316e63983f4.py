import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1)
        self.layer1 = self.build_layer(32, 32, 3, stride=1)
        self.layer2 = self.build_layer(32, 64, 3, stride=2)
        self.layer3 = self.build_layer(64, 128, 3, stride=2)
        self.layer4 = self.build_layer(128, 128, 3, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = None

    def build_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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
        self.criteria = self.criteria.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

def check_shape(shape):
    assert len(shape) == 4
    assert shape[1] in {3, 1}
    assert all(dim % 2 == 0 for dim in shape[2:])

def get_workspace_size(shape):
    check_shape(shape)
    net = Net(shape, (10,), {'lr': 0.01, 'momentum': 0.9}, None)
    return get_total_params(net)

def get_total_params(model):
    return sum(p.numel() for p in model.parameters())

def train_func(net, train_data, prm):
    net.train_setup(prm)
    net.learn(train_data)
    return net.state_dict()

def export_model():
    shape = (1, 3, 32, 32)
    net = Net(shape, (10,), {'lr': 0.01, 'momentum': 0.9}, None)
    workspace_size = get_workspace_size(shape)
    assert workspace_size <= 500000, workspace_size
    model = Net(shape, (10,), {'lr': 0.01, 'momentum': 0.9}, None)
    trained_weights = train_func(model, [], {'lr': 0.01, 'momentum': 0.9})
    return trained_weights