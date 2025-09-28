# Auto-generated single-file for PatchSplit
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from typing import Union
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def get(self, layer_type):
        if layer_type == 'LayerNorm':
            return nn.LayerNorm
        return None
from typing import Dict
import inspect
from typing import Tuple

# ---- mmengine.utils.dl_utils.parrots_wrapper.TORCH_VERSION ----
TORCH_VERSION = torch.__version__

# ---- mmengine.utils.dl_utils.parrots_wrapper._get_norm ----
def _get_norm() -> tuple:
    """A wrapper to obtain base classes of normalization layers from PyTorch or
    Parrots."""
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_

# ---- mmengine.utils.dl_utils.parrots_wrapper._BatchNorm ----
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()

# ---- mmengine.utils.dl_utils.parrots_wrapper._InstanceNorm ----
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()

# ---- mmcv.cnn.bricks.norm.infer_abbr ----
def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'

# ---- mmcv.cnn.bricks.norm.build_norm_layer ----
def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        with MODELS.switch_scope_and_registry(None) as registry:
            norm_layer = registry.get(layer_type)
        if norm_layer is None:
            raise KeyError(f'Cannot find {norm_layer} in registry under '
                           f'scope name {registry.scope}')
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

# ---- PatchSplit (target) ----
class PatchSplit(nn.Module):
    """The up-sample module used in neck (transformer pyramid network)

    Args:
        dim (int): the input dimension (channel number).
        fpn_dim (int): the fpn dimension (channel number).
        norm_cfg (dict): Config dict for normalization layer.
                Defaults to ``dict(type='LN')``.
    """

    def __init__(self, dim, fpn_dim, norm_cfg):
        super().__init__()
        _, self.norm = build_norm_layer(norm_cfg, dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(B, N, H, W, 2, 2,
                      self.fpn_dim).permute(0, 1, 2, 4, 3, 5,
                                            6).reshape(B, N, 2 * H, 2 * W,
                                                       self.fpn_dim)
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.patch_split = PatchSplit(
            dim=32,
            fpn_dim=16,
            norm_cfg={'type': 'LayerNorm'}
        )
        
        self.classifier = nn.Linear(16, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 2)
        x = self.patch_split(x)
        x = x.mean(dim=(1, 2, 3))
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
    
    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
