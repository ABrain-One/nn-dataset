# Auto-generated single-file for AttentionExtract
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# fnmatch
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
from typing import List, Optional, Union
import re
from typing import List
from typing import Union
from typing import Optional

# ---- original imports from contributing modules ----
from typing import Union, Optional, List
import fnmatch

# ---- AttentionExtract (target) ----
class AttentionExtract(torch.nn.Module):
    # defaults should cover a significant number of timm models with attention maps.
    default_node_names = ['*attn.softmax']
    default_module_names = ['*attn_drop']

    def __init__(
            self,
            model: Union[torch.nn.Module],
            names: Optional[List[str]] = None,
            mode: str = 'eval',
            method: str = 'fx',
            hook_type: str = 'forward',
            use_regex: bool = False,
    ):
        """ Extract attention maps (or other activations) from a model by name.

        Args:
            model: Instantiated model to extract from.
            names: List of concrete or wildcard names to extract. Names are nodes for fx and modules for hooks.
            mode: 'train' or 'eval' model mode.
            method: 'fx' or 'hook' extraction method.
            hook_type: 'forward' or 'forward_pre' hooks used.
            use_regex: Use regex instead of fnmatch
        """
        super().__init__()
        assert mode in ('train', 'eval')
        if mode == 'train':
            model = model.train()
        else:
            model = model.eval()

        assert method in ('fx', 'hook')
        if method == 'fx':
            # Simplified version without timm dependency
            self.model = model
            self.hooks = None
            matched = []
        else:
            # Simplified version without timm dependency
            self.model = model
            self.hooks = None
            matched = []

        self.names = matched
        self.mode = mode
        self.method = method

    def forward(self, x):
        return self.model(x)


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
        self.classifier = torch.nn.Linear(self._last_channels, self.num_classes)

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

        self.attention_extract = AttentionExtract(torch.nn.Identity())
        
        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
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
