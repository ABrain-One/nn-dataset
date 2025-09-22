# Auto-generated single-file for DropBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- mmdet.models.layers.dropblock.eps ----
eps = 1e-6

# ---- DropBlock (target) ----
class DropBlock(nn.Module):
    """Randomly drop some regions of feature maps.

     Please refer to the method proposed in `DropBlock
     <https://arxiv.org/abs/1810.12890>`_ for details.

    Args:
        drop_prob (float): The probability of dropping each block.
        block_size (int): The size of dropped blocks.
        warmup_iters (int): The drop probability will linearly increase
            from `0` to `drop_prob` during the first `warmup_iters` iterations.
            Default: 2000.
    """

    def __init__(self, drop_prob, block_size, warmup_iters=2000, **kwargs):
        super(DropBlock, self).__init__()
        assert block_size % 2 == 1
        assert 0 < drop_prob <= 1
        assert warmup_iters >= 0
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.warmup_iters = warmup_iters
        self.iter_cnt = 0

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map on which some areas will be randomly
                dropped.

        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return x
        self.iter_cnt += 1
        N, C, H, W = list(x.shape)
        gamma = self._compute_gamma((H, W))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(
            input=mask,
            stride=(1, 1),
            kernel_size=(self.block_size, self.block_size),
            padding=self.block_size // 2)
        mask = 1 - mask
        x = x * mask * mask.numel() / (eps + mask.sum())
        return x

    def _compute_gamma(self, feat_size):
        """Compute the value of gamma according to paper. gamma is the
        parameter of bernoulli distribution, which controls the number of
        features to drop.

        gamma = (drop_prob * fm_area) / (drop_area * keep_area)

        Args:
            feat_size (tuple[int, int]): The height and width of feature map.

        Returns:
            float: The value of gamma.
        """
        gamma = (self.drop_prob * feat_size[0] * feat_size[1])
        gamma /= ((feat_size[0] - self.block_size + 1) *
                  (feat_size[1] - self.block_size + 1))
        gamma /= (self.block_size**2)
        factor = (1.0 if self.iter_cnt > self.warmup_iters else self.iter_cnt /
                  self.warmup_iters)
        return gamma * factor

    def extra_repr(self):
        return (f'drop_prob={self.drop_prob}, block_size={self.block_size}, '
                f'warmup_iters={self.warmup_iters}')

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
        
        self.dropblock = DropBlock(
            drop_prob=0.1,
            block_size=3,
            warmup_iters=2000
        )

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
        x = self.dropblock(x)
        x = self.avgpool(x)
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
