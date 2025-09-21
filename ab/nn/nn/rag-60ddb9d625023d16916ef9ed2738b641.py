# Auto-generated single-file for CenterCropToSequence
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# transforms
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import math
from collections.abc import Sequence
from typing import List
import numbers
from typing import Union
from typing import Optional
from typing import Tuple
import torchvision.transforms as transforms

# ---- timm.data.naflex_transforms.get_image_size_for_seq ----
def get_image_size_for_seq(
        image_hw: Tuple[int, int],
        patch_size: Union[int, Tuple[int, int]] = 16,
        max_seq_len: int = 1024,
        divisible_by_patch: bool = True,
        max_ratio: Optional[float] = None,
        eps: float = 1e-5,
) -> Tuple[float, Tuple[int, int]]:
    """Determine scaling ratio and image size for sequence length constraint.

    Calculates the scaling ratio needed so that when image_hw is scaled,
    the total number of resulting patches does not exceed max_seq_len.

    Args:
        image_hw: Original image dimensions (height, width).
        patch_size: Patch dimensions. If int, patches are square.
        max_seq_len: Maximum allowed sequence length.
        divisible_by_patch: Whether resulting dimensions must be divisible by patch_size.
        max_ratio: Optional cap on scaling ratio to prevent excessive upsampling.
        eps: Convergence threshold for binary search.

    Returns:
        Tuple of (ratio, target_hw) where ratio is the scaling factor and
        target_hw is the resulting (height, width) after scaling.
    """

    # Handle patch size input, extract patch_h, patch_w
    if isinstance(patch_size, int):
        patch_h, patch_w = patch_size, patch_size
    else:
        # Assume it's a tuple/list: (patch_h, patch_w)
        if len(patch_size) != 2:
            raise ValueError("patch_size tuple must have exactly two elements (patch_h, patch_w).")
        patch_h, patch_w = patch_size

    # Safety checks
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_size dimensions must be positive.")

    def prepare_target_hw(ratio):
        """Scale image_hw by ratio and optionally round dimensions to multiples of patch_h, patch_w."""
        scaled_h = image_hw[0] * ratio
        scaled_w = image_hw[1] * ratio

        # If we need the result to be divisible by patch_size
        if divisible_by_patch:
            scaled_h = patch_h * math.ceil(scaled_h / patch_h)
            scaled_w = patch_w * math.ceil(scaled_w / patch_w)

        # Ensure at least one patch in each dimension
        scaled_h = int(max(scaled_h, patch_h))
        scaled_w = int(max(scaled_w, patch_w))

        return scaled_h, scaled_w

    def is_feasible(ratio):
        """Check if scaling by 'ratio' keeps patch count within max_seq_len."""
        t_h, t_w = prepare_target_hw(ratio)

        # Each dimension is already a multiple of patch_h, patch_w if divisible_by_patch=True.
        # Use integer division to count patches.
        num_patches_h = t_h // patch_h
        num_patches_w = t_w // patch_w
        seq_len = num_patches_h * num_patches_w

        return seq_len <= max_seq_len

    # Binary search boundaries
    lb = eps / 10.0
    rb = 100.0

    # Standard binary search loop
    while (rb - lb) >= eps:
        mid = (lb + rb) / 2.0
        if is_feasible(mid):
            lb = mid
        else:
            rb = mid

    # The final ratio from the binary search
    ratio = lb

    # If max_ratio is provided, clamp it to prevent upsampling beyond that threshold
    if max_ratio is not None:
        ratio = min(ratio, max_ratio)

    # Final checks
    if ratio <= eps:
        raise ValueError("Binary search failed - image might be too large?")
    if ratio >= 100.0:
        raise ValueError("Binary search failed - image might be too small?")

    # Prepare the final target dimensions with the possibly clamped ratio
    target_hw = prepare_target_hw(ratio)
    return ratio, target_hw

# ---- timm.data.transforms._setup_size ----
def _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

# ---- timm.data.transforms.center_crop_or_pad ----
def center_crop_or_pad(
        img: torch.Tensor,
        output_size: Union[int, List[int]],
        fill: Union[int, Tuple[int, int, int]] = 0,
        padding_mode: str = 'constant',
) -> torch.Tensor:
    """Center crops and/or pads the given image.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    output_size = _setup_size(output_size)
    crop_height, crop_width = output_size
    _, image_height, image_width = F.get_dimensions(img)

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill, padding_mode=padding_mode)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)

# ---- CenterCropToSequence (target) ----
class CenterCropToSequence(torch.nn.Module):
    """Center crop the image such that the resulting patch sequence length meets constraints."""
    def __init__(
            self,
            patch_size: int,
            max_seq_len: int,
            divisible_by_patch: bool = True,
            fill: Union[int, Tuple[int, int, int]] = 0,
            padding_mode: str = 'constant'
        ):
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.divisible_by_patch = divisible_by_patch
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """Center crop the image to maintain aspect ratio and fit sequence constraint."""
        _, h, w = transforms.functional.get_dimensions(img)
        _, target_hw = get_image_size_for_seq(
            (h, w),
            self.patch_size,
            self.max_seq_len,
            self.divisible_by_patch
        )

        # Use center crop
        return center_crop_or_pad(img, target_hw, fill=self.fill, padding_mode=self.padding_mode)

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.center_crop_to_seq = CenterCropToSequence(patch_size=16, max_seq_len=256)
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
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
