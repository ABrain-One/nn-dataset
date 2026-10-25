import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

    def fuse(self):
        # Fused version of Block for optimization purposes
        fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            bias=True
        ).requires_grad_(False)

        fused_conv.weight.copy_(self.conv.weight)
        if self.conv.bias is not None:
            fused_conv.bias.copy_(self.conv.bias)

        scripted_block = torch.jit.script(Block(
            in_channels=fused_conv.in_channels,
            out_channels=fused_conv.out_channels,
            kernel_size=fused_conv.kernel_size,
            stride=fused_conv.stride,
            padding=fused_conv.padding
        ).eval())
        scripted_block.batchnorm.running_mean.copy_(self.batchnorm.running_mean)
        scripted_block.batchnorm.running_var.copy_(self.batchnorm.running_var)
        scripted_block.batchnorm.weight.copy_(self.batchnorm.weight)
        scripted_block.batchnorm.bias.copy_(self.batchnorm.bias)

        return scripted_block

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # Follow the provided constraint: use in_shape as (batch, channels, height, width)
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 32, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(32, 64, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Block(64, 128, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(128 * 6 * 6, num_classes)
        self._set_training_mode(prm)

    def _set_training_mode(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        params = self.parameters()
        self.optimizer = torch.optim.SGD(
            params,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self._set_training_mode(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

# Auto-generated template for NerfLens, represents a unique class for image recognition.
class NerfLens(nn.Module):
    def __init__(self, in_module, prm):
        super(NerfLens, _s=4). __init__()
        self.in_module = in_module # .features
        self.fuse_blocks = nn.ModuleDict({f'b{i}': Block(in_channels, out_channels).fuse() for i, (in_channels, out_channels) in enumerate(zip(in_channels_list[:-1], in_channels_list[1:]))})
        self.prm_view_dependence = prm.get('view_dependence', False)
        self.view_dependent_bias = nn.Parameter(torch.zeros(1), requires_grad=False) if self.prm_view_dependence else None

    def forward(self, x):
        x = self.in_module(x)
        if self.view_dependent_bias is not None:
            x = x + self.view_dependent_bias.view(1, -1, 1, 1)
        for block in self.fuse_blocks.values():
            x = block(x)
        return x

    def fuse(self):
        fused_module = NerfLens(in_module=self.in_module, prm={'view_dependence': self.prm_view_dependence})
        for name, block in self.fuse_blocks.items():
            setattr(fused_module, name, block)
        return fused_module

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def _get_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k in self.state_dict()}

    def _set_from_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    print(f"Warning: unable to load {name}. Error: {e}")

    def load_pretrained(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self._set_from_state_dict(state_dict)