# Auto-generated single-file for DotProductAttentionLayer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- original imports from contributing modules ----

# ---- DotProductAttentionLayer (target) ----
class DotProductAttentionLayer(nn.Module):

    def __init__(self, dim_model=None):
        super().__init__()

        self.scale = dim_model**-0.5 if dim_model is not None else 1.

    def forward(self, query, key, value, mask=None):
        n, seq_len = mask.size()
        logits = torch.matmul(query.permute(0, 2, 1), key) * self.scale

        if mask is not None:
            mask = mask.view(n, 1, seq_len)
            logits = logits.masked_fill(mask, float('-inf'))

        weights = F.softmax(logits, dim=2)

        glimpse = torch.matmul(weights, value.transpose(1, 2))

        glimpse = glimpse.permute(0, 2, 1).contiguous()

        return glimpse

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        self.dot_product_attention = DotProductAttentionLayer(dim_model=32)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
