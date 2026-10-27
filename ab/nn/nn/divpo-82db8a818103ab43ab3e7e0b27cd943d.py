import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
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
            Block(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self._make_layer(32, 32, 3, 1, 1)
        self.layer2 = self._make_layer(32, 64, 3, 2, 1)
        self.layer3 = self._make_layer(64, 128, 3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride, groups):
        layers = [Block(in_channels, out_channels, stride=stride, groups=groups)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
            self.optimizer.step()

# Dummy input to derive the correct in_channels and out_shape
dummy_input = torch.randn(1, 3, 32, 32)
in_channels = dummy_input.shape[1]
out_shape = (10,)

# Parameters within the parameter limit
prm = {
    'lr': 0.01,
    'momentum': 0.9,
    'dropout': 0.2
}

# Device for training (e.g., "cuda" or "cpu")
device = "cpu"

# Initialize the model for export
model = Net(in_shape=(in_channels, 32, 32), out_shape=out_shape, prm=prm, device=device)

# Exported module properties
model.vocab_size = 10  # for the classifier
model.vocab = None  # no text components
model.prm = prm
model.dtype = torch.float32
model.device = device
model.max_tokens = None  # or any constant that makes sense in your model
model.dim = 32
model.dim_out = 10
model.weight_decay = prm.get('weight_decay', 0.0)
model.spec_dim = 32
model.n_heads = 1
model.n_layers = 1
model.n_embd = 32
model.max_positions = 32
model.pad_token_id = 0
model.tie_weights = False
model.positional_embedding = None
model.encoder_layer = None
model.encoder = None

# Ensure that all methods are defined and that the model does not require an initialization method (bind!)
model.learn = model.learn