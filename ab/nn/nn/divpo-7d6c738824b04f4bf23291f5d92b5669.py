import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = Block(32, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            Block(32, 64, kernel_size=3, stride=1, padding=1, groups=32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = Block(64, 64, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            Block(64, 128, kernel_size=3, stride=1, padding=1, groups=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
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
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Dummy input to derive the correct in_channels and out_channels.
in_channels_dummy = 3
in_height_dummy = 32
in_width_dummy = 32
out_channels_dummy = 10
dummy_input = torch.randn(1, in_channels_dummy, in_height_dummy, in_width_dummy)

# Benchmark parameters (these will be the maximum allowed).
batch_size = 1
channels = in_channels_dummy
height = in_height_dummy
width = in_width_dummy
params = 500000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Derive the correct in_channels based on the dummy input shape.
in_channels = dummy_input.shape[1]

# Calculate the total number of parameters.
def compute_total_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Target function to optimize: get the model to fit within params with as few layers as possible.
def faster_rcnn(in_shape, out_shape, prm):
    net = Net(in_shape, out_shape, prm, device)
    current_params = compute_total_params(net)
    while current_params > params:
        for child_name, child_module in net.named_children():
            if "layer" in child_name:
                # Remove the last layer and update the total params count.
                net._modules.pop(child_name)
                current_params = compute_total_params(net)
                break
        else:
            # If no layers were removed, try reducing the number of filters in the blocks.
            for block in net.modules():
                if isinstance(block, Block):
                    original_out_channels = block.conv.out_channels
                    new_out_channels = max(int(original_out_channels * 0.75), 8)
                    block.conv = nn.Conv2d(block.conv.in_channels, new_out_channels, kernel_size=3, padding=1, bias=False)
                    block.bn = nn.BatchNorm2d(new_out_channels)
                    current_params = compute_total_params(net)
                    break
            break
    return net

# Entry point to the optimization routine.
def train_and_export(in_shape, out_shape, prm):
    net = faster_rcnn(in_shape, out_shape, prm)
    data = torch.randn(batch_size, channels, height, width)
    net.learn([(data, torch.randint(0, out_channels_dummy, (data.shape[0],)))])
    exporter = nn.Sequential(
        net,
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((6, 6)),
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, out_channels_dummy)
    )
    exporter.to(device)
    exporter.eval()
    exporter.to(device)
    return exporter