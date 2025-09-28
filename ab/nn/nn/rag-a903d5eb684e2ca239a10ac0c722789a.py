# Auto-generated single-file for JumpingKnowledge
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# LSTM, Linear
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
from torch import Tensor
from typing import List, Optional
from typing import List
from typing import Optional

# ---- original imports from contributing modules ----
from torch.nn import LSTM, Linear

# ---- JumpingKnowledge (target) ----
class JumpingKnowledge(torch.nn.Module):
    r"""The Jumping Knowledge layer aggregation module from the
    `"Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper.

    Jumping knowledge is performed based on either **concatenation**
    (:obj:`"cat"`)

    .. math::

        \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)},

    **max pooling** (:obj:`"max"`)

    .. math::

        \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right),

    or **weighted summation**

    .. math::

        \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

    with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
    LSTM (:obj:`"lstm"`).

    Args:
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels (int, optional): The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
    """
    def __init__(
        self,
        mode: str,
        channels: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(channels, (num_layers * channels) // 2,
                             bidirectional=True, batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
            self.channels = channels
            self.num_layers = num_layers
        else:
            self.lstm = None
            self.att = None
            self.channels = None
            self.num_layers = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.att is not None:
            self.att.reset_parameters()

    def forward(self, xs: List[Tensor]) -> Tensor:
        r"""Forward pass.

        Args:
            xs (List[torch.Tensor]): List containing the layer-wise
                representations.
        """
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        else:  # self.mode == 'lstm'
            assert self.lstm is not None and self.att is not None
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)

    def __repr__(self) -> str:
        if self.mode == 'lstm':
            return (f'{self.__class__.__name__}({self.mode}, '
                    f'channels={self.channels}, layers={self.num_layers})')
        return f'{self.__class__.__name__}({self.mode})'

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
        self.jumping_knowledge = JumpingKnowledge(mode='cat', channels=32, num_layers=3)
        self.classifier = torch.nn.Linear(96, self.num_classes)

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
        
        xs = [x, x, x]
        x_combined = self.jumping_knowledge(xs)
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
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
