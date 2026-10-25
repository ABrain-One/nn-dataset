import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly latency
        if not self.training:
            self.conv = nn.utils.fuse_conv_bn_eval(self.conv, self.bn)
            del self.bn
            self.__dict__.pop('bn', None)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])

        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64),
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            nn.MaxPool2d(2, 2),
            Block(128, 256),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

        self._prepare_for_deployment(prm)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _prepare_for_deployment(self, prm):
        # Edge-friendly latency and parameter optimizations
        self.fuse_modules()
        if prm.get('fuse_activations', True):
            self.features[-1].relu = nn.Identity()

    def fuse_modules(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.fuse()

    def to_onnx(self, path):
        # Generate a dummy input for export
        dummy_input = torch.randn(1, self.in_channels, 32, 32, device=self.device)
        # Export to ONNX
        torch.onnx.export(self, dummy_input, path, opset_version=11, input_names=['input'], output_names=['output'])

    def __repr__(self):
        return f'EdgeCIFAR10(in_channels={self.in_channels}, num_classes={self.num_classes}, params={self._get_num_params()})'

    def _get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)