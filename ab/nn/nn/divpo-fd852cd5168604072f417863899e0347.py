import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused weight to reduce params
        fused_weight = torch.nn.functional.pad(self.conv.weight.squeeze(0), (1, 1, 1, 1))
        fused_weight = fused_weight.reshape(1, -1, self.conv.kernel_size[0], self.conv.kernel_size[1])
        fused_conv = Block(in_channels=self.conv.in_channels,
                           out_channels=self.conv.out_channels,
                           kernel_size=self.conv.kernel_size,
                           stride=self.conv.stride,
                           padding=self.conv.padding)
        fused_conv.conv.weight.data = fused_weight
        return fused_conv

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape, don't hardcode
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=2),
            Block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(256, 512, stride=2),
            Block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)
        self.to(device)
        self._set_optimizer(prm)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _set_optimizer(self, prm):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0005)
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def train_setup(self, prm):
        self.to(self.device)
        self._set_optimizer(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

# Edge-friendly pre-fuse block for parameter budget
class FusedBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fused_conv = self.fuse()

    def forward(self, x):
        return self.relu(self.batchnorm(self.fused_conv(x)))

    def train_setup(self, prm):
        self.to(self.device)
        self._set_optimizer(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def fuse(self):
        # Implement the fusion logic here
        # This is a placeholder for the actual fusion logic
        # You should replace this with the actual fusion logic you want to use
        fused_conv = super().fuse()
        return fused_conv

    def supported_hyperparameters(self):
        return {'lr', 'momentum', 'weight_decay'}

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding})"

    def export(self):
        # Placeholder for export logic
        # You should implement the actual export logic here
        return {"model": self}