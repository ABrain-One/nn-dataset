import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_description(self):
        return f"{self.__class__.__name__}(blocks={self._get_block_count()})"

    def _get_block_count(self):
        blocks = 0
        for _layer in self.children():
            blocks += len(list(_layer.children()))
        return blocks
    def get_flops(self, input_shape: tuple) -> int:
        input_size = (1, *input_shape)
        x = torch.rand(input_size, dtype=next(self.parameters()).dtype, device=self.device)
        flops = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tmp'),
            with_stack=True
        )(self)(x).size()
        return flops.numel()

    def _get_block_count(self):
        blocks = 0
        for _layer in self.children():
            blocks += len(list(_layer.children()))
        return blocks
    def _create_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool) -> nn.Module:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    def _make_layers(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool, blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(blocks):
            layers.append(self._create_block(in_channels, out_channels, kernel_size, stride, padding, bias))
            in_channels = out_channels
            stride = 1
        return nn.Sequential(*layers)

    def _make_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool) -> nn.Module:
        return self._make_layers(in_channels, out_channels, kernel_size, stride, padding, bias, blocks=2)

    def _make_layer(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool, blocks: int) -> nn.Module:
        layers = []
        for _ in range(blocks):
            layers.append(self._create_block(in_channels, out_channels, kernel_size, stride, padding, bias))
            in_channels = out_channels
            stride = 1
        return nn.Sequential(*layers)
    def _create_conv_bn(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def _create_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool) -> nn.Module:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    def _create_bn(self, in_channels: int) -> nn.BatchNorm2d:
        return nn.BatchNorm2d(in_channels)
    def _create_act(self) -> nn.ReLU:
        return nn.ReLU(inplace=True)