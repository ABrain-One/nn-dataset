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

    def fuse(self):
        # Fuse conv, bn, and activation for better efficiency
        if hasattr(self.conv, 'weight_fused'):
            return
        fused_weight = self.conv.weight * self.bn.weight.view(-1, 1, 1, 1)
        if self.conv.bias is None:
            bias = self.bn.bias
        else:
            bias = self.conv.bias + self.bn.bias
        self.conv.weight_fused = fused_weight
        self.conv.bias_fused = bias
        self.conv.in_channels_fused = self.conv.in_channels
        self.conv.out_channels_fused = self.conv.out_channels
        self.conv.kernel_size_fused = self.conv.kernel_size
        self.conv.stride_fused = self.conv.stride
        self.conv.padding_fused = self.conv.padding
        self.conv.dilation_fused = self.conv.dilation
        self.conv.groups_fused = self.conv.groups
        self.conv.padding_mode_fused = self.conv.padding_mode
        self.conv.transposed_fused = self.conv.transposed
        self.conv.output_padding_fused = self.conv.output_padding
        del self.bn
        self.relu = None

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # handle in_shape as (batch, channels, height, width)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2 = Block(32, 64, kernel_size=3, stride=2, padding=1)
        self.block3 = Block(64, 128, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.block2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Dummy output to ensure the model is well-formed
model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm={'lr': 0.01, 'momentum': 0.9}, device='cpu')