# Auto-generated single-file for SimpleRoIAlign
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union
import collections
from itertools import repeat
from typing import Union
from collections import *
from typing import Tuple

# ---- mmcv.ops.point_sample.normalize ----
def normalize(grid: Tensor) -> Tensor:
    """Normalize input grid from [-1, 1] to [0, 1]

    Args:
        grid (torch.Tensor): The grid to be normalize, range [-1, 1].

    Returns:
        torch.Tensor: Normalized grid, range [0, 1].
    """

    return (grid + 1.0) / 2.0

# ---- mmcv.ops.point_sample.generate_grid ----
def generate_grid(num_grid: int, size: Tuple[int, int],
                  device: torch.device) -> Tensor:
    """Generate regular square grid of points in [0, 1] x [0, 1] coordinate
    space.

    Args:
        num_grid (int): The number of grids to sample, one for each region.
        size (tuple[int, int]): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        torch.Tensor: A tensor of shape (num_grid, size[0]*size[1], 2) that
        contains coordinates for the regular grids.
    """

    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device)
    grid = F.affine_grid(
        affine_trans, torch.Size((1, 1, *size)), align_corners=False)
    grid = normalize(grid)
    return grid.view(1, -1, 2).expand(num_grid, -1, -1)

# ---- mmcv.ops.point_sample.denormalize ----
def denormalize(grid: Tensor) -> Tensor:
    """Denormalize input grid from range [0, 1] to [-1, 1]

    Args:
        grid (torch.Tensor): The grid to be denormalize, range [0, 1].

    Returns:
        torch.Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0

# ---- mmcv.ops.point_sample.point_sample ----
def point_sample(input: Tensor,
                 points: Tensor,
                 align_corners: bool = False,
                 **kwargs) -> Tensor:
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (torch.Tensor): Feature map, shape (N, C, H, W).
        points (torch.Tensor): Image based absolute point coordinates
            (normalized), range [0, 1] x [0, 1], shape (N, P, 2) or
            (N, Hgrid, Wgrid, 2).
        align_corners (bool, optional): Whether align_corners.
            Default: False

    Returns:
        torch.Tensor: Features of `point` on `input`, shape (N, C, P) or
        (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

# ---- mmcv.ops.point_sample.get_shape_from_feature_map ----
def get_shape_from_feature_map(x: Tensor) -> Tensor:
    """Get spatial resolution of input feature map considering exporting to
    onnx mode.

    Args:
        x (torch.Tensor): Input tensor, shape (N, C, H, W)

    Returns:
        torch.Tensor: Spatial resolution (width, height), shape (1, 1, 2)
    """
    img_shape = torch.tensor(x.shape[2:]).flip(0).view(1, 1,
                                                       2).to(x.device).float()
    return img_shape

# ---- mmcv.ops.point_sample.abs_img_point_to_rel_img_point ----
def abs_img_point_to_rel_img_point(abs_img_points: Tensor,
                                   img: Union[tuple, Tensor],
                                   spatial_scale: float = 1.) -> Tensor:
    """Convert image based absolute point coordinates to image based relative
    coordinates for sampling.

    Args:
        abs_img_points (torch.Tensor): Image based absolute point coordinates,
            shape (N, P, 2)
        img (tuple or torch.Tensor): (height, width) of image or feature map.
        spatial_scale (float, optional): Scale points by this factor.
            Default: 1.

    Returns:
        Tensor: Image based relative point coordinates for sampling, shape
        (N, P, 2).
    """

    assert (isinstance(img, tuple) and len(img) == 2) or \
           (isinstance(img, torch.Tensor) and len(img.shape) == 4)

    if isinstance(img, tuple):
        h, w = img
        scale = torch.tensor([w, h],
                             dtype=torch.float,
                             device=abs_img_points.device)
        scale = scale.view(1, 1, 2)
    else:
        scale = get_shape_from_feature_map(img)

    return abs_img_points / scale * spatial_scale

# ---- mmcv.ops.point_sample.rel_roi_point_to_abs_img_point ----
def rel_roi_point_to_abs_img_point(rois: Tensor,
                                   rel_roi_points: Tensor) -> Tensor:
    """Convert roi based relative point coordinates to image based absolute
    point coordinates.

    Args:
        rois (torch.Tensor): RoIs or BBoxes, shape (N, 4) or (N, 5)
        rel_roi_points (torch.Tensor): Point coordinates inside RoI, relative
            to RoI, location, range (0, 1), shape (N, P, 2)
    Returns:
        torch.Tensor: Image based absolute point coordinates, shape (N, P, 2)
    """

    with torch.no_grad():
        assert rel_roi_points.size(0) == rois.size(0)
        assert rois.dim() == 2
        assert rel_roi_points.dim() == 3
        assert rel_roi_points.size(2) == 2
        # remove batch idx
        if rois.size(1) == 5:
            rois = rois[:, 1:]
        abs_img_points = rel_roi_points.clone()
        # To avoid an error during exporting to onnx use independent
        # variables instead inplace computation
        xs = abs_img_points[:, :, 0] * (rois[:, None, 2] - rois[:, None, 0])
        ys = abs_img_points[:, :, 1] * (rois[:, None, 3] - rois[:, None, 1])
        xs += rois[:, None, 0]
        ys += rois[:, None, 1]
        abs_img_points = torch.stack([xs, ys], dim=2)
    return abs_img_points

# ---- mmcv.ops.point_sample.rel_roi_point_to_rel_img_point ----
def rel_roi_point_to_rel_img_point(rois: Tensor,
                                   rel_roi_points: Tensor,
                                   img: Union[tuple, Tensor],
                                   spatial_scale: float = 1.) -> Tensor:
    """Convert roi based relative point coordinates to image based absolute
    point coordinates.

    Args:
        rois (torch.Tensor): RoIs or BBoxes, shape (N, 4) or (N, 5)
        rel_roi_points (torch.Tensor): Point coordinates inside RoI, relative
            to RoI, location, range (0, 1), shape (N, P, 2)
        img (tuple or torch.Tensor): (height, width) of image or feature map.
        spatial_scale (float, optional): Scale points by this factor.
            Default: 1.

    Returns:
        torch.Tensor: Image based relative point coordinates for sampling,
        shape (N, P, 2).
    """

    abs_img_point = rel_roi_point_to_abs_img_point(rois, rel_roi_points)
    rel_img_point = abs_img_point_to_rel_img_point(abs_img_point, img,
                                                   spatial_scale)

    return rel_img_point

# ---- torch.nn.modules.utils._ntuple ----
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

# ---- torch.nn.modules.utils._pair ----
_pair = _ntuple(2, "_pair")

# ---- SimpleRoIAlign (target) ----
class SimpleRoIAlign(nn.Module):

    def __init__(self,
                 output_size: Tuple[int],
                 spatial_scale: float,
                 aligned: bool = True) -> None:
        """Simple RoI align in PointRend, faster than standard RoIAlign.

        Args:
            output_size (tuple[int]): h, w
            spatial_scale (float): scale the input boxes by this number
            aligned (bool): if False, use the legacy implementation in
                MMDetection, align_corners=True will be used in F.grid_sample.
                If True, align the results more perfectly.
        """

        super().__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        # to be consistent with other RoI ops
        self.use_torchvision = False
        self.aligned = aligned

    def forward(self, features: Tensor, rois: Tensor) -> Tensor:
        num_imgs = features.size(0)
        num_rois = rois.size(0)
        rel_roi_points = generate_grid(
            num_rois, self.output_size, device=rois.device)

        point_feats = []
        for batch_ind in range(num_imgs):
            # unravel batch dim
            feat = features[batch_ind].unsqueeze(0)
            inds = (rois[:, 0].long() == batch_ind)
            if inds.any():
                rel_img_points = rel_roi_point_to_rel_img_point(
                    rois[inds], rel_roi_points[inds], feat,
                    self.spatial_scale).unsqueeze(0)
                point_feat = point_sample(
                    feat, rel_img_points, align_corners=not self.aligned)
                point_feat = point_feat.squeeze(0).transpose(0, 1)
                point_feats.append(point_feat)

        point_feats_t = torch.cat(point_feats, dim=0)

        channels = features.size(1)
        roi_feats = point_feats_t.reshape(num_rois, channels,
                                          *self.output_size)

        return roi_feats

    def __repr__(self) -> str:
        format_str = self.__class__.__name__
        format_str += '(output_size={}, spatial_scale={}'.format(
            self.output_size, self.spatial_scale)
        return format_str

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
        self.roi_align = SimpleRoIAlign(output_size=(7, 7), spatial_scale=1.0, aligned=True)
        self.classifier = nn.Linear(64 * 7 * 7, self.num_classes)

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
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        rois = torch.zeros(batch_size, 5, device=x.device)
        rois[:, 0] = torch.arange(batch_size, device=x.device) 
        rois[:, 1] = 0 
        rois[:, 2] = 0 
        rois[:, 3] = width  
        rois[:, 4] = height 
        roi_features = self.roi_align(x, rois)
        roi_features = roi_features.view(roi_features.size(0), -1)
        x = self.classifier(roi_features)
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
