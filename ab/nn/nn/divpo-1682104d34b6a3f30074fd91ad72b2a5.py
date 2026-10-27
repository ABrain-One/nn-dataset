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
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]
        self.stem = nn.Sequential(
            Block(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = Block(64, 64, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = Block(128, 128, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            Block(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self.to(device)
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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Dummy input to derive the correct in_channels and out_shape.
dummy_input = torch.randn(1, 3, 32, 32)
in_channels = dummy_input.shape[1]
out_shape = (10,)

# Parameters within the budget and the required total size.
max_params = 500000

# Calculate the estimated total parameter count.
def estimate_params(model):
    return sum(p.numel() for p in model.parameters())

# Design function to be implemented.
def design_model(in_shape, out_shape, prm):
    device = torch.device('cpu')
    total_params = estimate_params(Net(in_shape, out_shape, prm, device))

    # Design parameters:
    for _ in range(5):
        batch_size = in_shape[0]
        channels = in_shape[1]
        height = in_shape[2]
        width = in_shape[3]
        num_classes = out_shape[0]

        # Fixed parameters:
        layers = 3
        groups = 1
        upscale = 2

        # Varied parameters:
        stem_out = 32
        stem_k = 3
        stem_s = 1
        stem_p = (stem_k - 1) // 2
        early_out = 32
        early_k = 3
        early_s = 1
        early_p = (early_k - 1) // 2
        mid_out = 64
        mid_k = 3
        mid_s = 1
        mid_p = (mid_k - 1) // 2
        late_out = 128
        late_k = 3
        late_s = 1
        late_p = (late_k - 1) // 2

        model = Net(
            in_shape,
            out_shape,
            prm,
            device
        )
        model.stem = nn.Sequential(
            Block(channels, stem_out, stem_k, stem_s, stem_p, groups),
            nn.MaxPool2d(upscale, upscale)
        )
        model.layer1 = Block(stem_out, early_out, early_k, early_s, early_p)
        model.layer2 = nn.Sequential(
            Block(early_out, mid_out, mid_k, mid_s, mid_p, groups),
            nn.MaxPool2d(upscale, upscale)
        )
        model.layer3 = Block(mid_out, late_out, late_k, late_s, late_p)
        model.layer4 = nn.Sequential(
            Block(late_out, late_out, late_k, late_s, late_p, groups),
            nn.MaxPool2d(upscale, upscale)
        )
        model.avgpool = nn.AdaptiveAvgPool2d((height // upscale, width // upscale))

        total_params = estimate_params(model)
        if total_params > max_params:
            continue

        # Classifier based on the provided out_shape and in_channels.
        model.classifier = nn.Linear(late_out * (height // upscale) * (width // upscale), num_classes)

        return model

# Run the design function to generate the model.
model = design_model(in_shape=dummy_input.shape, out_shape=out_shape, prm={'lr': 0.01, 'momentum': 0.9})