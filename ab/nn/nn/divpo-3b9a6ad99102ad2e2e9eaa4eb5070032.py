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
        self.block1 = self._make_block(32, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )

    def _make_block(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


def supported_hyperparameters():
    return {'lr', 'momentum'}


def check_trace(model_in, model_out, x):
    model = model_in(in_shape=(3, 224, 224), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device=torch.device("cpu"))
    model.train()
    model.forward = model_out
    model.to(torch.device("cpu"))

    tracer = torch.fx.symbolic_trace(model)
    assert isinstance(tracer, torch.fx.GraphModule)
    assert tracer.code == model.forward.code_template
    assert tracer.code_template

    traced_script_module = torch.jit.trace(model, x)
    assert traced_script_module is not None
    assert traced_script_module(x).shape[0] == x.shape[0]

def test_check_trace():
    class Net1(nn.Module):
        def __init__(self, in_shape, out_shape, prm, device):
            super().__init__()
            self.in_channels = in_shape[1]
            self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32 * 55 * 55, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, out_shape[0])

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def code_template(self):
            return "x = F.relu(self.conv1(x)); x = self.pool(x); x = torch.flatten(x, 1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x)"

    class Net2(nn.Module):
        def __init__(self, in_shape, out_shape, prm, device):
            super().__init__()
            self.in_channels = in_shape[1]
            self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32 * 55 * 55, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, out_shape[0])

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def code_template(self):
            return "x = F.relu(self.conv1(x)); x = self.pool(x); x = torch.flatten(x, 1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = self.fc3(x)"

    x = torch.randn(1, 3, 224, 224)
    check_trace(Net1, Net2, x)



model = Net((3, 32, 32), (10,), {'lr': 0.01, 'momentum': 0.9}, torch.device("cpu"))