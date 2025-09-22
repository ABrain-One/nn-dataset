# Auto-generated single-file for CrossMapLRN2d
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# Function, _cross_map_lrn2d
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
from torch.nn import Module
from torch import Tensor
from torch.autograd import Function

# ---- _cross_map_lrn2d ----
class _cross_map_lrn2d(Function):
    @staticmethod
    def forward(ctx, input, size, alpha, beta, k):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

# ---- CrossMapLRN2d (target) ----
class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.cross_map_lrn = CrossMapLRN2d(size=5)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.cross_map_lrn(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
