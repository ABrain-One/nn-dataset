# Auto-generated single-file for QFormer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from typing import Callable

# ---- original imports from contributing modules ----

# ---- QFormer (target) ----
class QFormer(torch.nn.Module):
    r"""The Querying Transformer (Q-Former) from
    `"BLIP-2: Bootstrapping Language-Image Pre-training
    with Frozen Image Encoders and Large Language Models"
    <https://arxiv.org/pdf/2301.12597>`_ paper.

    Args:
        input_dim (int): The number of features in the input.
        hidden_dim (int): The dimension of the fnn in the encoder layer.
        output_dim (int): The final output dimension.
        num_heads (int): The number of multi-attention-heads.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        dropout (int): The dropout value in each encoder layer.

    .. note::
        This is a simplified version of the original Q-Former implementation.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.0,
            activation: Callable = torch.nn.ReLU(),
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )
        self.project = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Input sequence to the encoder layer.
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, sequence length :math:`N`,
                and feature dimension :math:`F`.
        """
        x = self.layer_norm(x)
        x = self.encoder(x)
        out = self.project(x)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_heads={self.num_heads}, '
                f'num_layers={self.num_layers})')

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
        self.qformer = QFormer(input_dim=32, hidden_dim=64, output_dim=32, num_heads=4, num_layers=2, dropout=0.1)
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.qformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
