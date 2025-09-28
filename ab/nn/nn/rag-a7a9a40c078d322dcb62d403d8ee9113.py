# Auto-generated single-file for Encoding
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- original imports from contributing modules ----
from torch import nn

# ---- Encoding (target) ----
class Encoding(nn.Module):
    """Encoding Layer: a learnable residual encoder.

    Input is of shape  (batch_size, channels, height, width).
    Output is of shape (batch_size, num_codes, channels).

    Args:
        channels: dimension of the features or feature channels
        num_codes: number of code words
    """

    def __init__(self, channels, num_codes):
        super().__init__()
        # init codewords and smoothing factor
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, channels,
                        dtype=torch.float).uniform_(-std, std),
            requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(
            torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0),
            requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)

        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        # [batch_size, channels, height, width]
        batch_size = x.size(0)
        # [batch_size, height x width, channels]
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), dim=2)
        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}' \
                    f'x{self.channels})'
        return repr_str

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
        self.encoding = Encoding(channels=32, num_codes=8)
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
        x = F.adaptive_avg_pool2d(x, (8, 8))
        encoded_feat = self.encoding(x)
        x = encoded_feat.mean(dim=1)
        
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
