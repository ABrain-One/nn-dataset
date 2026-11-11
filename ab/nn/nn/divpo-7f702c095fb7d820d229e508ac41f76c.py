import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBlock(in_channels, hidden_dim, kernel_size=1, bias=False)
            )
        layers.extend([
            ConvBlock(
                hidden_dim, hidden_dim, stride=stride, expand_padding=False,
                groups=hidden_dim, bias=False
            ),
            nn.Conv2d(
                hidden_dim, out_channels, kernel_size=1, padding=0,
                stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1,
            padding=None, expand_padding=True, groups=1, bias=True
    ):
        super().__init__()
        padding = padding or (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.out_channels = 32
        self.num_classes = out_shape[0]
        self.num_blocks = [3, 4, 6, 3]
        self.expand_ratios = [1, 6, 6, 6]
        self.block_sizes = [32, 64, 128, 256]
        self.stem_size = 32
        self.image_size = 32
        self.dropout = prm.get('dropout', 0.0)
        self.stem_channels = [32, 32, 32, 32, 64, 64, 64]
        self.input_size = (32, 32)
        self.stem_output_channels = [32, 64, 64, 64, 64, 64, 64]
        self.model_size = 32
        self.model_channels = 64
        self.depths = [3, 4, 6, 3]
        self.num_features = 64
        self.num_conv_layers = 5
        self.model_type = "MobileNetV3"

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stem_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.stem_channels[0]),
            nn.ReLU6(inplace=True)
        )

        inverted_residual_block = InvertedResidual
        prev_channels = self.stem_channels[0]
        for stage_id, block_info in enumerate(zip(self.block_sizes, self.num_blocks, self.expand_ratios)):
            block_size, num_blocks, expand_ratio = block_info
            for _ in range(num_blocks):
                channels = prev_channels
                stride = 2 if prev_channels == self.model_channels and stage_id < 3 else 1
                inverted_residual = inverted_residual_block(
                    channels, channels, stride=stride, expand_ratio=expand_ratio
                )
                self.features.add_module(
                    f"{stage_id}_{channels}_{channels}", inverted_residual
                )
                prev_channels = channels

        self.features.add_module(
            "final_pool", nn.AdaptiveAvgPool2d(1)
        )
        self.features.add_module(
            "flatten", nn.Flatten()
        )
        self.classifier = nn.Linear(prev_channels, self.num_classes)

        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        return self.classifier(self.features(x))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()