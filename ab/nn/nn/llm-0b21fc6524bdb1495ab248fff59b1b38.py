import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def _edge_backbone(name, device):
    import torchvision.models as _tvm
    ctor = getattr(_tvm, name)
    m = ctor(weights='DEFAULT')
    if name.startswith('mobilenet') or name.startswith('efficientnet') or name.startswith('squeezenet'):
        feat = m.features
    elif name.startswith('mnasnet'):
        feat = m.layers
    elif name.startswith('shufflenet'):
        feat = torch.nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5)
    elif name.startswith('regnet'):
        feat = torch.nn.Sequential(m.stem, m.trunk_output)
    else:
        feat = m.features
    return feat.to(device)

class OptimizedHead(nn.Module):

    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        p = float(prm['dropout'])
        self.reduce = nn.Conv2d(in_channels, 96, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(192)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(192, 128, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.relu1(self.bn_reduce(self.reduce(x)))
        x = self.relu2(self.bn1(self.conv1(x)))
        x = self.relu3(self.bn2(self.conv2(x)))
        x = self.relu4(self.bn3(self.conv3(x)))
        x = self.relu5(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.dropout = float(prm['dropout'])
        self.batch = int(prm['batch'])
        self.learning_rate = float(prm['lr'])
        self.momentum_value = float(prm['momentum'])
        self.num_epochs = int(prm['epoch'])
        self._transform_name = prm['transform']
        self.backbone = _edge_backbone('shufflenet_v2_x1_0', device)
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_shape[1]), int(in_shape[2]), int(in_shape[3]), device=device)
            feat = self.backbone(dummy)
        in_channels = int(feat.shape[1])
        self.head = OptimizedHead(in_channels, int(out_shape[0]), prm)
        self.to(device)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        lr = float(prm['lr'])
        momentum = float(prm['momentum'])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        batch_hint = self.batch
        for _epoch in range(max(1, self.num_epochs)):
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
                loss.backward()
                self.optimizer.step()