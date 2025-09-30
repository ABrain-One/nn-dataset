# Auto-generated single-file for CornerPool
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
import warnings
from packaging import version as pkg_version
parse = pkg_version.parse

# ---- mmcv.ops.corner_pool._corner_pool ----
def _corner_pool(x: Tensor, dim: int, flip: bool) -> Tensor:
    size = x.size(dim)
    output = x.clone()

    ind = 1
    while ind < size:
        if flip:
            cur_start = 0
            cur_len = size - ind
            next_start = ind
            next_len = size - ind
        else:
            cur_start = ind
            cur_len = size - ind
            next_start = 0
            next_len = size - ind

        # max_temp should be cloned for backward computation
        max_temp = output.narrow(dim, cur_start, cur_len).clone()
        cur_temp = output.narrow(dim, cur_start, cur_len)
        next_temp = output.narrow(dim, next_start, next_len)

        cur_temp[...] = torch.where(max_temp > next_temp, max_temp, next_temp)

        ind = ind << 1

    return output

# ---- mmengine.utils.version_utils.digit_version ----
def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)

# ---- CornerPool (target) ----
class CornerPool(nn.Module):
    """Corner Pooling.

    Corner Pooling is a new type of pooling layer that helps a
    convolutional network better localize corners of bounding boxes.

    Please refer to `CornerNet: Detecting Objects as Paired Keypoints
    <https://arxiv.org/abs/1808.01244>`_ for more details.

    Code is modified from https://github.com/princeton-vl/CornerNet-Lite.

    Args:
        mode (str): Pooling orientation for the pooling layer

            - 'bottom': Bottom Pooling
            - 'left': Left Pooling
            - 'right': Right Pooling
            - 'top': Top Pooling

    Returns:
        Feature map after pooling.
    """

    cummax_dim_flip = {
        'bottom': (2, False),
        'left': (3, True),
        'right': (3, False),
        'top': (2, True),
    }

    def __init__(self, mode: str):
        super().__init__()
        assert mode in self.cummax_dim_flip
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if (torch.__version__ != 'parrots' and
                digit_version(torch.__version__) >= digit_version('1.5.0')):
            dim, flip = self.cummax_dim_flip[self.mode]
            if flip:
                x = x.flip(dim)
            pool_tensor, _ = torch.cummax(x, dim=dim)
            if flip:
                pool_tensor = pool_tensor.flip(dim)
            return pool_tensor
        else:
            dim, flip = self.cummax_dim_flip[self.mode]
            return _corner_pool(x, dim, flip)

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CornerPool(mode='bottom'),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

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
