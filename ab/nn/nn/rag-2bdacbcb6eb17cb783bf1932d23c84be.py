# Auto-generated single-file for FilterResponseNormTlu2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
def _assert(condition, message): assert condition, message

# ---- timm.layers.filter_response_norm.inv_instance_rms ----
def inv_instance_rms(x, eps: float = 1e-5):
    rms = x.square().float().mean(dim=(2, 3), keepdim=True).add(eps).rsqrt().to(x.dtype)
    return rms.expand(x.shape)

# ---- FilterResponseNormTlu2d (target) ----
class FilterResponseNormTlu2d(nn.Module):
    def __init__(self, num_features, apply_act=True, eps=1e-5, rms=True, **_):
        super(FilterResponseNormTlu2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.tau = nn.Parameter(torch.zeros(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = (1, -1, 1, 1)
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)
        return torch.maximum(x, self.tau.reshape(v_shape).to(dtype=x_dtype)) if self.tau is not None else x

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
        self.filter_response_norm_tlu = FilterResponseNormTlu2d(num_features=32, apply_act=True)
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
        x = self.filter_response_norm_tlu(x)
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
