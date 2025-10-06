# Auto-generated single-file for EvoNorm2dB0
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
def _assert(condition, message): assert condition, message

# ---- timm.layers.evo_norm.instance_std ----
def instance_std(x, eps: float = 1e-5):
    std = x.float().var(dim=(2, 3), unbiased=False, keepdim=True).add(eps).sqrt().to(x.dtype)
    return std.expand(x.shape)

# ---- EvoNorm2dB0 (target) ----
class EvoNorm2dB0(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-3, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        if self.v is not None:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                # var = manual_var(x, dim=(0, 2, 3)).squeeze()
                n = x.numel() / x.shape[1]
                self.running_var.copy_(
                    self.running_var * (1 - self.momentum) +
                    var.detach() * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            left = var.add(self.eps).sqrt_().to(x_dtype).view(v_shape).expand_as(x)
            v = self.v.to(x_dtype).view(v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.to(x_dtype).view(v_shape) + self.bias.to(x_dtype).view(v_shape)

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
        self.evo_norm = EvoNorm2dB0(num_features=32, apply_act=True, momentum=0.1, eps=1e-3)
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
        x = self.evo_norm(x)
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
