# Auto-generated single-file for BottleneckBlock
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# comm, differentiable_all_reduce, dist, env, weight_init
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn import LayerNorm

# Mock missing dependencies
def differentiable_all_reduce(x):
    return x

class MockDist:
    def get_world_size(self):
        return 1

dist = MockDist()

class MockEnv:
    TORCH_VERSION = (1, 5)

env = MockEnv()

class MockComm:
    def get_world_size(self):
        return 1

comm = MockComm()

# ---- detectron2.layers.batch_norm.FrozenBatchNorm2d ----
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
        """
        Convert all FrozenBatchNorm2d to BatchNorm2d

        Args:
            module (torch.nn.Module):

        Returns:
            If module is FrozenBatchNorm2d, returns a new module.
            Otherwise, in-place convert module and return it.

        This is needed for quantization:
            https://fb.workplace.com/groups/1043663463248667/permalink/1296330057982005/
        """

        res = module
        if isinstance(module, FrozenBatchNorm2d):
            res = torch.nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

# ---- detectron2.layers.blocks.CNNBlockBase ----
class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

# ---- detectron2.utils.env.TORCH_VERSION ----
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

# ---- detectron2.layers.wrappers.check_if_dynamo_compiling ----
def check_if_dynamo_compiling():
    if TORCH_VERSION >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False

# ---- detectron2.layers.wrappers.Conv2d ----
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# ---- detectron2.layers.wrappers.BatchNorm2d ----
BatchNorm2d = torch.nn.BatchNorm2d

# ---- detectron2.layers.batch_norm.NaiveSyncBatchNorm ----
class NaiveSyncBatchNorm(BatchNorm2d):
    """
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
    """

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if comm.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        half_input = input.dtype == torch.float16
        if half_input:
            # fp16 does not have good enough numerics for the reduction here
            input = input.float()
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == "":
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
                vec = vec + input.sum()  # make sure there is gradient w.r.t input
            else:
                vec = torch.cat(
                    [
                        mean,
                        meansqr,
                        torch.ones([1], device=mean.device, dtype=mean.dtype),
                    ],
                    dim=0,
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clamp(max=1) * self.momentum  # no update if total_batch is 0
            mean, meansqr, _ = torch.split(vec / total_batch.clamp(min=1), C)  # avoid div-by-zero

        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        ret = input * scale + bias
        if half_input:
            ret = ret.half()
        return ret

# ---- detectron2.layers.batch_norm.get_norm ----
def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)

# ---- detectron2.layers.wrappers.Conv2dNormActivation ----
class Conv2dNormActivation(nn.Sequential):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        norm_layer=None,
        activation_layer=None,
        dilation=1,
        bias=None,
    ):
        layers = [
            Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=(norm_layer is None) if bias is None else bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer)
        super().__init__(*layers)

# ---- BottleneckBlock (target) ----
class BottleneckBlock(nn.Module):
    """Slightly modified BottleNeck block (extra relu and biases)"""

    def __init__(self, in_channels, out_channels, *, norm_layer, stride=1):
        super().__init__()

        # See note in ResidualBlock for the reason behind bias=True
        self.convnormrelu1 = Conv2dNormActivation(
            in_channels, out_channels // 4, norm_layer=norm_layer, kernel_size=1, bias=True
        )
        self.convnormrelu2 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, norm_layer=norm_layer, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.convnormrelu3 = Conv2dNormActivation(
            out_channels // 4, out_channels, norm_layer=norm_layer, kernel_size=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = Conv2dNormActivation(
                in_channels,
                out_channels,
                norm_layer=norm_layer,
                kernel_size=1,
                stride=stride,
                bias=True,
                activation_layer=None,
            )

    def forward(self, x):
        y = x
        y = self.convnormrelu1(y)
        y = self.convnormrelu2(y)
        y = self.convnormrelu3(y)

        x = self.downsample(x)

        return self.relu(x + y)


def supported_hyperparameters():
    return {'lr','momentum'}


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
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self.bottleneck_block = BottleneckBlock(
            in_channels=32,
            out_channels=32,
            norm_layer=torch.nn.BatchNorm2d,
            stride=1
        )

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.bottleneck_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
