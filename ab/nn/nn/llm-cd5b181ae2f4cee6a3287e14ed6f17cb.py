import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def supported_hyperparameters():
    return {'lr', 'momentum'}

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

class Net(nn.Module):

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=float(prm['lr']), momentum=float(prm['momentum']))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.backbone = _edge_backbone('mnasnet1_0', device)
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_shape[1]), int(in_shape[2]), int(in_shape[3]), device=device)
            feat = self.backbone(dummy)
        in_channels = int(feat.shape[1])
        self.head = nn.Sequential(nn.Dropout(p=0.13316094561637615), nn.Linear(in_channels, int(out_shape[0])))
        self.to(device)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.head(x)
        return x