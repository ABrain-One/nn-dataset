# Auto-generated single-file for PatchMerge
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import math
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    @staticmethod
    def get(layer_type):
        if layer_type == 'LayerNorm':
            return nn.LayerNorm
        return None

# ---- original imports from contributing modules ----

# ---- mmpretrain.models.utils.norm.build_norm_layer ----
def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.

        num_features (int): Number of input channels.

    Returns:
        nn.Module: The created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = MODELS.get(layer_type)
    if norm_layer is None:
        raise KeyError(f'Cannot find {layer_type} in registry under scope '
                       f'name {MODELS.scope}')

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        layer = norm_layer(num_channels=num_features, **cfg_)

    if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
        layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer

# ---- PatchMerge (target) ----
class PatchMerge(nn.Module):
    """PatchMerge for HiViT.

    Args:
        dim (int): Number of input channels.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self, dim, norm_cfg):
        super().__init__()
        self.norm = build_norm_layer(norm_cfg, dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, *args, **kwargs):
        is_main_stage = len(x.shape) == 3
        if is_main_stage:
            B, N, C = x.shape
            S = int(math.sqrt(N))
            x = x.reshape(B, S // 2, 2, S // 2, 2, C) \
                .permute(0, 1, 3, 2, 4, 5) \
                .reshape(B, -1, 2, 2, C)
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)

        if is_main_stage:
            x = x[:, :, 0, 0, :]
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
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.patch_merge = PatchMerge(
            dim=32,
            norm_cfg={'type': 'LayerNorm'}
        )
        
        self.classifier = nn.Linear(64, self.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        x = self.patch_merge(x)
        x = x.mean(dim=1)
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
