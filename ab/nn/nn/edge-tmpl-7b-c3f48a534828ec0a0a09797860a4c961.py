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
        self.projector = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.projector(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.avg_pool(x).flatten(1)
        return self.drop(x)
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
        self.backbone = _edge_backbone('regnet_x_400mf', device)
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_shape[1]), int(in_shape[2]), int(in_shape[3]), device=device)
            feat = self.backbone(dummy)
        in_channels = int(feat.shape[1])
        self.head = Head(in_channels, int(out_shape[0]), prm)
        self.to(device)

# ============ LLM-GENERATED forward / train_setup / learn ============
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm['lr']))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criteria(self(inputs), labels)
            loss.backward()
            self.optimizer.step()
# =====================================================================