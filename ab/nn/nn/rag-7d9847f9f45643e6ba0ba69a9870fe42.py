# Auto-generated single-file for AutoAugment
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# magnitude_id, op_name, p
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from enum import Enum
import math
from torchvision.transforms import InterpolationMode

# ---- torchvision.transforms.autoaugment.AutoAugmentPolicy ----
class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """

    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"

# ---- torchvision.transforms.autoaugment._apply_op ----
def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[list[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

# ---- AutoAugment (target) ----
class AutoAugment(torch.nn.Module):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> list[tuple[tuple[str, float, Optional[int]], tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError(f"The provided policy {policy} is not recognized.")

    def _augmentation_space(self, num_bins: int, image_size: tuple[int, int]) -> dict[str, tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

    def get_params(transform_num: int) -> tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.policies))

        op_meta = self._augmentation_space(10, (height, width))
        for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(policy={self.policy}, fill={self.fill})"


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self.auto_augment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        
        layers += [
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
