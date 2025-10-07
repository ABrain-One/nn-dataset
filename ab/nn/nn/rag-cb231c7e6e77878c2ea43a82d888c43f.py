# Auto-generated single-file for GeneralizedMeanPooling
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmpretrain.models.necks.gem.gem ----
def gem(x: Tensor, p: Parameter, eps: float = 1e-6, clamp=True) -> Tensor:
    if clamp:
        x = x.clamp(min=eps)
    return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

# ---- GeneralizedMeanPooling (target) ----
class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        p (float): Parameter value. Defaults to 3.
        eps (float): epsilon. Defaults to 1e-6.
        clamp (bool): Use clamp before pooling. Defaults to True
        p_trainable (bool): Toggle whether Parameter p is trainable or not.
            Defaults to True.
    """

    def __init__(self, p=3., eps=1e-6, clamp=True, p_trainable=True):
        assert p >= 1, "'p' must be a value greater than 1"
        super(GeneralizedMeanPooling, self).__init__()
        self.p = Parameter(torch.ones(1) * p, requires_grad=p_trainable)
        self.eps = eps
        self.clamp = clamp
        self.p_trainable = p_trainable

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([
                gem(x, p=self.p, eps=self.eps, clamp=self.clamp)
                for x in inputs
            ])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = gem(inputs, p=self.p, eps=self.eps, clamp=self.clamp)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

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
        self.gem_pooling = GeneralizedMeanPooling(p=3.0, eps=1e-6, clamp=True, p_trainable=True)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.gem_pooling(x)
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
