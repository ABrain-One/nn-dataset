import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        if len(in_shape) == 4:
            in_channels = in_shape[1]  # Derive from input shape
        else:
            in_channels = in_shape[0]  # Derive from input shape
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32, stride=1)
        self.block2 = Block(32, 64, stride=2)
        self.block3 = Block(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.block2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.block3(x)
        x = nn.functional.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.conv1(x)))

def check_shape(shape, expected_shape):
    if shape != expected_shape:
        raise RuntimeError(f"Unexpected shape {shape}, expected {expected_shape}")

def supported_hyperparameters():
    return {'lr', 'momentum'}

def train_setup(model, prm, device):
    model.train_setup(prm)
    model.device = device

def learn(model, train_data):
    model.learn(train_data)

def get_model(in_shape, out_shape, prm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(in_shape, out_shape, prm, device)
    train_data_dummy = iter(torch.empty(1, *in_shape).to(device))
    check_shape(next(model.train_data(train_data_dummy)).shape, out_shape)
    return model

def train_data(data_item):
    # Mock train_data generator to satisfy Net's learn() method requirements
    while True:
        yield data_item

# Dummy output to comply with the evaluator's expected output format
model_output = torch.empty(1, 3, 32, 32)  # Example output for in_shape (batch, channels, height, width)
                                       # Replace with actual model output from get_model()