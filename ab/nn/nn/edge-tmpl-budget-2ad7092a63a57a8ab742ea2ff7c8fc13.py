import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# hyperparameters used below: 'lr', 'momentum', 'epoch', 'batch', 'dropout', 'transform'
def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

def _edge_backbone(name, device):
    """Return (feature_extractor, ) for a torchvision backbone with ImageNet
    weights; only the conv feature portion is kept (classifier dropped)."""
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
# ===================== LLM-GENERATED HEAD =====================
class Head(nn.Module):
    def __init__(self, in_channels, num_classes, prm):
        super().__init__()
        p = float(prm['dropout'])
        self.reduce_dim = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.reduce_dim(x)))
        x = self.dw_conv(x)
        x = self.bn2(x)
        se_weights = self.se_block(x)
        x = x * se_weights
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))
# =============================================================

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
        self.backbone = _edge_backbone('squeezenet1_0', device)
        # freeze BN stats stability not required; measure feature channels
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_shape[1]), int(in_shape[2]), int(in_shape[3]), device=device)
            feat = self.backbone(dummy)
        in_channels = int(feat.shape[1])
        self.head = Head(in_channels, int(out_shape[0]), prm)
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
        batch_hint = self.batch  # 'batch' honored by the external DataLoader
        for _epoch in range(max(1, self.num_epochs)):
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
                loss.backward()
                self.optimizer.step()