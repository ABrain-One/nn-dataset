import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (B, C, H, W) — derived from in_shape
        num_classes = int(out_shape[0])  # number of target classes
        
        # First layer takes in_channels from in_shape[1]
        self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature extractor
        self.features = nn.Sequential(
            self._make_block(16, 16, 3, 3, 1, 2),
            self._make_block(16, 32, 3, 3, 1, 2),
            self._make_block(32, 32, 3, 3, 1, 1),
            self._make_block(32, 64, 3, 3, 1, 2),
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        
        # Optional dropout
        p = prm.get('dropout', 0.0)
        if p > 0:
            self.dropout = nn.Dropout(p)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout:
            x = self.dropout(x)
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

    @staticmethod
    def _make_block(in_channels, out_channels, kernel_size, stride, pad, num_blocks):
        block = nn.Sequential()
        block.add_module(f"conv_{in_channels}_{kernel_size}", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        block.add_module(f"bn_{out_channels}", nn.BatchNorm2d(out_channels))
        block.add_module(f"relu_{out_channels}", nn.ReLU(inplace=True))
        for i in range(1, num_blocks):
            block.add_module(f"conv_{out_channels}_{kernel_size}_{i}", nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False))
            block.add_module(f"bn_{out_channels}_{i}", nn.BatchNorm2d(out_channels))
            block.add_module(f"relu_{out_channels}_{i}", nn.ReLU(inplace=True))
        return block