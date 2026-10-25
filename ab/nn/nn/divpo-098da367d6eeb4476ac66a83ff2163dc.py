import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly deployment
        if isinstance(self.relu, nn.ReLU) and isinstance(self.batchnorm, nn.BatchNorm2d):
            fused_conv = nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                bias=True
            )
            fused_conv.weight.data = (self.batchnorm.weight.data / self.batchnorm.running_var.sqrt()) * self.conv.weight.data
            if not torch.isnan(fused_conv.weight.data).any():
                fused_conv.weight.data += (self.batchnorm.bias.data - self.batchnorm.running_mean.data) * self.batchnorm.weight.data / self.batchnorm.running_var.sqrt()
                fused_conv.bias.data = self.conv.bias.data
                self.fused_layer = fused_conv
                del self.conv
                del self.batchnorm
                del self.relu
        elif self.fused_layer is not None:
            # Already fused
            pass
        else:
            # Fallback to unfused block for training/CPU compatibility
            self.conv = nn.Conv2d(
                in_channels=self.conv.in_channels,
                out_channels=self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding
            )
            self.batchnorm = nn.BatchNorm2d(self.conv.out_channels)
            self.relu = nn.ReLU(inplace=True)

    def fuse_all_layers(self):
        # Apply fusion to the block and any nested blocks
        self.fuse()
        if hasattr(self, 'children') and self.children:
            for child_block in self.children:
                child_block.fuse_all_layers()

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle dataset-specific shape quirks
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)

        self.features = nn.Sequential(
            Block(self.in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fuse_target = None  # for fusion tuning
        self.classifier = nn.Linear(128 * 4 * 4, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', self.learning_rate),
            momentum=prm.get('momentum', self.momentum),
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

    def fuse_blocks(self, target_block=None):
        # Fuse all blocks or a specific target block for edge optimization
        if target_block is None:
            for block in self.features:
                block.fuse_all_layers()
        else:
            target_block.fuse_all_layers()
        if target_block:
            self.fuse_target = target_block
        self.to(self.device)