import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def fuse(self):
        # Fused block for edge-friendly latency
        assert self.training is False
        for m in self.children():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        self.conv.fuse_bias_and_weights()
        fused_block = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(self.conv.out_channels),
            nn.ReLU(inplace=True)
        )
        return fused_block

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # Follow the constraint for edge-friendly handling
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = Block(32, 64, stride=1)
        self.layer2 = Block(64, 128, stride=2)
        self.layer3 = Block(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.to(device)

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=0.0001)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        out = self.classifier(x)
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
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

    def summary(self):
        return str(self)

    def to_onnx(self, path):
        input_shape = (1, self.in_channels, 32, 32)
        x = torch.rand(*input_shape)
        output_path = f"{path}/model.onnx"
        torch.onnx.export(self, x, output_path, export_params=True, opset_version=11, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
        print(f"ONNX model exported to {output_path}")

    def __str__(self):
        return f"Net(in_channels={self.in_channels}, out_channels=[64, 128, 256], num_classes={self.classifier.out_features})"