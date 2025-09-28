# Auto-generated single-file for MultilabelClassificationHead
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
from collections.abc import Sequence
import torch

# ---- mmengine.model.weight_init.normal_init ----
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# ---- mmpose.models.heads.heatmap_heads.internet_head.make_linear_layers ----
def make_linear_layers(feat_dims, relu_final=False):
    """Make linear layers."""
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or \
                (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# ---- MultilabelClassificationHead (target) ----
class MultilabelClassificationHead(nn.Module):
    """MultilabelClassificationHead is a sub-module of Interhand3DHead, and
    outputs hand type classification.

    Args:
        in_channels (int): Number of input channels. Defaults to 2048.
        num_labels (int): Number of labels. Defaults to 2.
        hidden_dims (Sequence[int]): Number of hidden dimension of FC layers.
            Defaults to ``(512, )``.
    """

    def __init__(self,
                 in_channels: int = 2048,
                 num_labels: int = 2,
                 hidden_dims: Sequence[int] = (512, )):

        super().__init__()

        self.in_channels = in_channels

        feature_dims = [in_channels, *hidden_dims, num_labels]
        self.fc = make_linear_layers(feature_dims, relu_final=False)

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, x):
        """Forward function."""
        labels = self.fc(x)
        return labels

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
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.multilabel_classification_head = MultilabelClassificationHead(
            in_channels=32,
            num_labels=self.num_classes,
            hidden_dims=(64,)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.multilabel_classification_head(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
