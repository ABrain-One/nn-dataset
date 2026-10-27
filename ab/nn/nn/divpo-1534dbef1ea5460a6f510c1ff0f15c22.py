import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
    
    def learn(self, train_data: torch.utils.data.DataLoader) -> None:
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

# This code is part of a template. Do NOT assign any values to `prm` unless it's one of the supported keys.
# For example, you can assign `prm = {'lr': value}` when `'lr' in supported_hyperparameters()`.
prm = {}

# REQUIRED: Generate a unique class definition below.
# Start with `class Net(nn.Module):` and implement entirely within this block.
# Use `in_shape` and `out_shape` to derive the first convolutional layer's input channels and output channels.
# The class must define `__init__` and `forward` methods, and MAY implement `forward` and `training_setup` methods.
# You MAY NOT use `nn.AdaptiveAvgPool2d`, `nn.AdaptiveAvgPool1d`, `nn.AdaptiveMaxPool1d`, `nn.AdaptiveMaxPool2d`, `nn.AvgPool1d`, `nn.AvgPool2d`, `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.LPPool1d`, `nn.LPPool2d`, `nn.MaxPool3d`, `nn.AdaptiveMaxPool3d`, `nn.MaxPool3d` in any form.
# All layers must support `fused_dropout` (see `torch.nn.functional.dropout`), and you MUST use `dropout` as a parameter (no hardcoded dropout rates).
# Parameters:
#   - `in_shape`: (C, H, W) channels-first shape of the input tensor
#   - `out_shape`: (N,) class indices tensor shape
#   - `prm`: a dictionary with the following keys:
#     - `'lr'`: a float value indicating the learning rate
#     - `'momentum'`: a float value indicating the momentum
#     - `'dropout'`: a float value indicating the dropout rate (if present)
#   - `device`: a torch.device object indicating the device to use
# The class must define a `learn` method that accepts a DataLoader and trains the model in-place.
# The class MUST NOT use any data outside the given DataLoader for training or any other purpose.
# The class MUST NOT use any data outside the given DataLoader for training or any other purpose.
# The class MAY define a `training_setup` method with the following signature:
#   def training_setup(self, prm: dict) -> None: ...
# This method MAY NOT be present, and MUST NOT be called by the evaluator.
# The method MAY perform any setup that does not require data from the DataLoader.
# The method MAY perform any setup that does not require data from the DataLoader.
# The method MAY be present but MUST NOT be used by the model definition code.
# The method MAY be present but MUST NOT be used by the model definition code.
# The class MAY implement the `forward` method with the following signature:
#   def forward(self, x: torch.Tensor) -> torch.Tensor: ...
# The method MUST NOT call `super().forward(x)`.
# The method MAY NOT be present and MUST be implemented by the model class.
# The method MAY NOT be present and MUST be implemented by the model class.
# The method MUST NOT be used by the model class in its `forward` method.
# The method MUST NOT be used by the model class in its `forward` method.
# The method MUST be designed to process an input tensor of shape (B, C, H, W) and return an output tensor of shape (B, N).
# The method MUST NOT be used by the model class in its `forward` method.
# The method MUST NOT be used by the model class in its `forward` method.
# The method MUST be designed to process an input tensor of shape (B, C, H, W) and return an output tensor of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY NOT be present and MUST NOT be used by the model class in its `forward` method.
# The method MAY NOT be present and MUST NOT be used by the model class in its `forward` method.
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N).
# The method MAY be present and MAY be used in the final model definition to process input tensors of shape (B, C, H, W) and return output tensors of shape (B, N