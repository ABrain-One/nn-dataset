from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torchvision.models._utils import _make_divisible
from torchvision.ops.misc import Conv2dNormActivation


class InvertedResidual(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):

    def train_setup(self, prm):
        """
          - The augment schedule is validated with renormalize policy.
          - The loss criterion uses reduction='none' so learn() can support both a scalar lam and a per-sample lam tensor.
       
        """
        self.to(self.device)
        # elementwise CE.
        self.criteria = (nn.CrossEntropyLoss(reduction='none').to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        # Cosine LR decay over the full run.
        # For 1-epoch proxy runs T_max=1 => constant lr.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(int(prm.get('epoch', 1)), 1))

        self.batch_transform_fn = None
        self.augment_configs = None

        if prm.get('augment'):
            try:
                from ab.nn.transform.base.BatchTransform import BatchTransform as bt_mod
                batch_transform = bt_mod.batch_transform
                validate_schedule = getattr(bt_mod, 'validate_schedule', None)
            except ImportError as e:
                raise RuntimeError(f"Failed to load batch_transform: {e}")


            config = prm.get('augment')
            if validate_schedule is not None:
                config, report = validate_schedule(config, policy="renormalize")
                if config is None:
                    raise ValueError(f"Invalid augment schedule: {report['reason']}")
                if report['repairs']:
                    print(f"Augment schedule repaired: {report['repairs']}")

            self.batch_transform_fn = batch_transform
            self.augment_configs = config
            print(f"Augmentation loaded: {self.augment_configs}")



    def learn(self, train_data):
        """
        loss_i = lam_i * CE(out_i, a_i) + (1 - lam_i) * CE(out_i, b_i).
        With lam = 1.0 and target_b = target_a, reduces to standard cross-entropy.
        """
        total_batches = len(train_data)
        ce = self.criteria[0]  # reduction='none' -> per-sample loss vector

        for batch_idx, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.batch_transform_fn:
                inputs, target_a, target_b, lam = self.batch_transform_fn(
                    inputs, labels, batch_idx, total_batches, self.augment_configs
                )
                outputs = self(inputs)
                if torch.is_tensor(lam):
                    lam = lam.to(self.device).float().clamp(0.0, 1.0)  # [B] or scalar tensor
                loss_vec = lam * ce(outputs, target_a) + (1.0 - lam) * ce(outputs, target_b)
                loss = loss_vec.mean()
            else:
                outputs = self(inputs)
                loss = ce(outputs, labels).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

        # one scheduler step per epoch.
        if getattr(self, 'scheduler', None) is not None:
            self.scheduler.step()



    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        num_classes: int = out_shape[0]
        width_mult: float = 1.0
        inverted_residual_setting: Optional[List[List[int]]] = None
        round_nearest: int = 8
        block: Optional[Callable[..., nn.Module]] = None
        norm_layer: Optional[Callable[..., nn.Module]] = None
        dropout: float = prm['dropout']
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(in_shape[1], input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)