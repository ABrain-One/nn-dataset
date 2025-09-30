# Auto-generated single-file for RandomCropToSequence
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import random
from typing import Union
from typing import Optional
from typing import Tuple
from torchvision.transforms import InterpolationMode
from torchvision import transforms

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

# ---- timm.data.transforms.crop_or_pad ----
def crop_or_pad(
        img: torch.Tensor,
        top: int,
        left: int,
        height: int,
        width: int,
        fill: Union[int, Tuple[int, int, int]] = 0,
        padding_mode: str = 'constant',
) -> torch.Tensor:
    """ Crops and/or pads image to meet target size, with control over fill and padding_mode.
    """
    if len(img.shape) == 4:
        _, _, image_height, image_width = img.shape
    else:
        _, image_height, image_width = img.shape
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(image_width, left), 0),
            max(bottom - max(image_height, top), 0),
        ]
        img = F.pad(img, padding_ltrb, fill=fill, padding_mode=padding_mode)

    top = max(top, 0)
    left = max(left, 0)
    return img[:, :, top:top+height, left:left+width]

# ---- RandomCropToSequence (target) ----
class RandomCropToSequence(torch.nn.Module):
    """Randomly crop and/or pad the image to fit sequence length constraints.

    This maintains aspect ratio while ensuring the resulting image, when divided into patches,
    will not exceed the specified maximum sequence length. Similar to CentralCropToSequence
    but with randomized positioning.
    """

    def __init__(
            self,
            patch_size: int,
            max_sequence_len: int,
            divisible_by_patch: bool = True,
            fill: Union[int, Tuple[int, int, int]] = 0,
            padding_mode: str = 'constant'
    ):
        """
        Args:
            patch_size: Size of patches (int or tuple of (patch_h, patch_w))
            max_sequence_len: Maximum allowed sequence length for the resulting image
            divisible_by_patch: If True, resulting image dimensions will be multiples of patch_size
            fill: Fill value for padding
            padding_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        """
        super().__init__()
        self.patch_size = patch_size
        self.max_sequence_len = max_sequence_len
        self.divisible_by_patch = divisible_by_patch
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, target_size):
        """Get random position for crop/pad."""
        if len(img.shape) == 4:
            _, _, image_height, image_width = img.shape
        else:
            _, image_height, image_width = img.shape
        delta_height = image_height - target_size[0]
        delta_width = image_width - target_size[1]

        # Handle both positive (crop) and negative (pad) deltas
        if delta_height == 0:
            top = 0
        else:
            top = int(math.copysign(random.randint(0, abs(delta_height)), delta_height))

        if delta_width == 0:
            left = 0
        else:
            left = int(math.copysign(random.randint(0, abs(delta_width)), delta_width))

        return top, left

    def forward(self, img):
        """Randomly crop or pad the image to maintain aspect ratio and fit sequence constraint."""
        # Get current dimensions
        if len(img.shape) == 4:
            _, _, img_h, img_w = img.shape
        else:
            _, img_h, img_w = img.shape

        # Calculate target dimensions that satisfy sequence length
        # We use max_ratio=1.0 to prevent upscaling - we only want to crop or maintain current size
        _, target_hw = get_image_size_for_seq(
            (img_h, img_w),
            self.patch_size,
            self.max_sequence_len,
            self.divisible_by_patch,
            max_ratio=1.0  # Prevent upscaling
        )

        # Get random position for crop/pad
        top, left = self.get_params(img, target_hw)

        # Apply crop or pad
        return crop_or_pad(
            img,
            top=top,
            left=left,
            height=target_hw[0],
            width=target_hw[1],
            fill=self.fill,
            padding_mode=self.padding_mode,
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(patch_size={self.patch_size}, "
                f"max_sequence_len={self.max_sequence_len}, "
                f"divisible_by_patch={self.divisible_by_patch})")

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
        self.random_crop_to_seq = RandomCropToSequence(patch_size=4, max_sequence_len=49)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.random_crop_to_seq(x)
        x = self.features(x)
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
