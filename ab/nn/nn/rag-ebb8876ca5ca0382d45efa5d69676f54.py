# Auto-generated single-file for RepConditionalPosEnc
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from typing import Union
from typing import Tuple
from typing import Optional

# ---- original imports from contributing modules ----

# ---- RepConditionalPosEnc (target) ----
class RepConditionalPosEnc(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
            inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            dim: Number of input channels.
            dim_out: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepConditionalPosEnc, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.dim = dim
        self.dim_out = dim_out or dim
        self.groups = dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim_out,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=spatial_shape[0] // 2,
                groups=self.groups,
                bias=True,
            )
        else:
            self.reparam_conv = None
            self.pos_enc = nn.Conv2d(
                self.dim,
                self.dim_out,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                groups=self.groups,
                bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
        else:
            x = self.pos_enc(x) + x
        return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.dim // self.groups
        kernel_value = torch.zeros(
            (
                self.dim,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pos_enc.weight.dtype,
            device=self.pos_enc.weight.device,
        )
        for i in range(self.dim):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pos_enc.weight
        b_final = self.pos_enc.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            self.dim,
            self.dim_out,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__("pos_enc")

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
        self.pos_enc = RepConditionalPosEnc(dim=64, dim_out=64, spatial_shape=(7, 7))
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.pos_enc(x)
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
