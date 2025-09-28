# Auto-generated single-file for FeatureMapProcessor
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
import warnings
from collections.abc import Sequence
from typing import List
from typing import Union
from typing import Optional
from typing import Tuple
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmpose.models.utils.ops.resize ----
def resize(input: torch.Tensor,
           size: Optional[Union[Tuple[int, int], torch.Size]] = None,
           scale_factor: Optional[float] = None,
           mode: str = 'nearest',
           align_corners: Optional[bool] = None,
           warning: bool = True) -> torch.Tensor:
    """Resize a given input tensor using specified size or scale_factor.

    Args:
        input (torch.Tensor): The input tensor to be resized.
        size (Optional[Union[Tuple[int, int], torch.Size]]): The desired
            output size. Defaults to None.
        scale_factor (Optional[float]): The scaling factor for resizing.
            Defaults to None.
        mode (str): The interpolation mode. Defaults to 'nearest'.
        align_corners (Optional[bool]): Determines whether to align the
            corners when using certain interpolation modes. Defaults to None.
        warning (bool): Whether to display a warning when the input and
            output sizes are not ideal for alignment. Defaults to True.

    Returns:
        torch.Tensor: The resized tensor.
    """
    # Check if a warning should be displayed regarding input and output sizes
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would be more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')

    # Convert torch.Size to tuple if necessary
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)

    # Perform the resizing operation
    return F.interpolate(input, size, scale_factor, mode, align_corners)

# ---- FeatureMapProcessor (target) ----
class FeatureMapProcessor(nn.Module):
    """A PyTorch module for selecting, concatenating, and rescaling feature
    maps.

    Args:
        select_index (Optional[Union[int, Tuple[int]]], optional): Index or
            indices of feature maps to select. Defaults to None, which means
            all feature maps are used.
        concat (bool, optional): Whether to concatenate the selected feature
            maps. Defaults to False.
        scale_factor (float, optional): The scaling factor to apply to the
            feature maps. Defaults to 1.0.
        apply_relu (bool, optional): Whether to apply ReLU on input feature
            maps. Defaults to False.
        align_corners (bool, optional): Whether to align corners when resizing
            the feature maps. Defaults to False.
    """

    def __init__(
        self,
        select_index: Optional[Union[int, Tuple[int]]] = None,
        concat: bool = False,
        scale_factor: float = 1.0,
        apply_relu: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()

        if isinstance(select_index, int):
            select_index = (select_index, )
        self.select_index = select_index
        self.concat = concat

        assert (
            scale_factor > 0
        ), f'the argument `scale_factor` must be positive, ' \
           f'but got {scale_factor}'
        self.scale_factor = scale_factor
        self.apply_relu = apply_relu
        self.align_corners = align_corners

    def forward(self, inputs: Union[Tensor, Sequence[Tensor]]
                ) -> Union[Tensor, List[Tensor]]:

        if not isinstance(inputs, (tuple, list)):
            sequential_input = False
            inputs = [inputs]
        else:
            sequential_input = True

            if self.select_index is not None:
                inputs = [inputs[i] for i in self.select_index]

            if self.concat:
                inputs = self._concat(inputs)

        if self.apply_relu:
            inputs = [F.relu(x) for x in inputs]

        if self.scale_factor != 1.0:
            inputs = self._rescale(inputs)

        if not sequential_input:
            inputs = inputs[0]

        return inputs

    def _concat(self, inputs: Sequence[Tensor]) -> List[Tensor]:
        size = inputs[0].shape[-2:]
        resized_inputs = [
            resize(
                x,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        return [torch.cat(resized_inputs, dim=1)]

    def _rescale(self, inputs: Sequence[Tensor]) -> List[Tensor]:
        rescaled_inputs = [
            resize(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=self.align_corners,
            ) for x in inputs
        ]
        return rescaled_inputs

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
        self.feature_map_processor = FeatureMapProcessor(select_index=0, concat=False, scale_factor=1.0, apply_relu=True)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.feature_map_processor(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
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
            self.optimizer.step()
