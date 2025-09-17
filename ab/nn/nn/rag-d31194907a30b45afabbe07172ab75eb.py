import torch
import torch.nn.functional as F

import torch, torch.nn as nn

class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, additional_context: int = 0):
        super(CausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        self.padding = (kernel_size - 1) * dilation - stride + 1

        if additional_context < 0:
            raise ValueError("additional_context must be non-negative")

        if additional_context > self.padding:
            raise ValueError("additional_context can't be greater than the padding")

        self.additional_context = additional_context

        self.left_padding = self.padding - additional_context

    # Input shape is (N, C_in, L_in)
    def forward(self, x: torch.Tensor):
        # Right padding is always zero, because think about it: during training, you don't know what happens AFTER the training sample
        # Padding with zeros is not a valid assumption, so you just would need to shorten the output length by that amount
        x = torch.nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)

    def streaming_forward(self, x: torch.Tensor, state: torch.Tensor):
        input = torch.cat((state, x), dim=2)

        result = self.conv(input)

        # Update the state
        state = input[:, :, result.shape[2] * self.conv.stride[0]:]

        return result, state


import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        # Stable 2D stem to avoid channel/shape mismatches
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # Example patterns you may use (choose ONE appropriate to the block):
        # layers += [CausalConv1d(in_ch=32, out_ch=32, kernel_size=3, stride=1, dilation=1)]

        # Keep under parameter budget and end with a known channel count:
        self._last_channels = 32
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()