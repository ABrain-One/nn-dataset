# Auto-generated single-file for MessageNorm
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional

# ---- original imports from contributing modules ----
from torch.nn import Parameter

# ---- MessageNorm (target) ----
class MessageNorm(torch.nn.Module):
    r"""Applies message normalization over the aggregated messages as described
    in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::

        \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_{i} + s \cdot
        {\| \mathbf{x}_i \|}_2 \cdot
        \frac{\mathbf{m}_{i}}{{\|\mathbf{m}_i\|}_2} \right)

    Args:
        learn_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor :math:`s` of message normalization.
            (default: :obj:`False`)
        device (torch.device, optional): The device to use for the module.
            (default: :obj:`None`)
    """
    def __init__(self, learn_scale: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.scale = Parameter(torch.empty(1, device=device),
                               requires_grad=learn_scale)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.scale.data.fill_(1.0)

    def forward(self, x: Tensor, msg: Tensor, p: float = 2.0) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            msg (torch.Tensor): The message tensor :math:`\mathbf{M}`.
            p (float, optional): The norm :math:`p` to use for normalization.
                (default: :obj:`2.0`)
        """
        msg = F.normalize(msg, p=p, dim=-1)
        x_norm = x.norm(p=p, dim=-1, keepdim=True)
        return msg * x_norm * self.scale

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'(learn_scale={self.scale.requires_grad})')

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.message_norm = MessageNorm(learn_scale=True, device=device)
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=False),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        msg = torch.tanh(x) * 0.5  
        x_normalized = self.message_norm(x, msg)
        x_combined = x + x_normalized
        return self.classifier(x_combined)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
