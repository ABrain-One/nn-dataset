# Auto-generated single-file for MaskLabel
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- MaskLabel (target) ----
class MaskLabel(torch.nn.Module):
    r"""The label embedding and masking layer from the `"Masked Label
    Prediction: Unified Message Passing Model for Semi-Supervised
    Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    Here, node labels :obj:`y` are merged to the initial node features :obj:`x`
    for a subset of their nodes according to :obj:`mask`.

    .. note::

        For an example of using :class:`MaskLabel`, see
        `examples/unimp_arxiv.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/unimp_arxiv.py>`_.

    Args:
        num_classes (int): The number of classes.
        out_channels (int): Size of each output sample.
        method (str, optional): If set to :obj:`"add"`, label embeddings are
            added to the input. If set to :obj:`"concat"`, label embeddings are
            concatenated. In case :obj:`method="add"`, then :obj:`out_channels`
            needs to be identical to the input dimensionality of node features.
            (default: :obj:`"add"`)
    """
    def __init__(self, num_classes: int, out_channels: int,
                 method: str = "add"):
        super().__init__()

        self.method = method
        if method not in ["add", "concat"]:
            raise ValueError(
                f"'method' must be either 'add' or 'concat' (got '{method}')")

        self.emb = torch.nn.Embedding(num_classes, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.emb.reset_parameters()

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """"""  # noqa: D419
        if self.method == "concat":
            out = x.new_zeros(y.size(0), self.emb.weight.size(-1))
            out[mask] = self.emb(y[mask])
            return torch.cat([x, out], dim=-1)
        else:
            x = torch.clone(x)
            x[mask] += self.emb(y[mask])
            return x

    def ratio_mask(mask: Tensor, ratio: float):
        r"""Modifies :obj:`mask` by setting :obj:`ratio` of :obj:`True`
        entries to :obj:`False`. Does not operate in-place.

        Args:
            mask (torch.Tensor): The mask to re-mask.
            ratio (float): The ratio of entries to keep.
        """
        n = int(mask.sum())
        out = mask.clone()
        out[mask] = torch.rand(n, device=mask.device) < ratio
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

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
        self.classifier = nn.Linear(32, self.num_classes)
        self.mask_label = MaskLabel(num_classes=self.num_classes, out_channels=32, method="add")

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
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        B = x.size(0)
        y = torch.randint(0, self.num_classes, (B,), device=x.device)
        mask = torch.ones(B, dtype=torch.bool, device=x.device)
        
        x = self.mask_label(x, y, mask)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()