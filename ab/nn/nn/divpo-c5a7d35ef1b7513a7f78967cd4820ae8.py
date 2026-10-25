import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        expand_channels = in_channels * expand_ratio
        layers.append(
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(expand_channels))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False)
        )
        layers.append(nn.BatchNorm2d(expand_channels))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]

        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # EfficientNet-like inverted residual blocks
        # Use an example block for reference, then adjust the block count for total parameters to not exceed 500000.
        # You can create your own block or use the provided ones to design the block structure.
        # Example block (for reference only; may not fit within the parameter budget):
        # class ExampleBlock(nn.Module):
        #     def __init__(self, in_channels, out_channels, stride):
        #         super(ExampleBlock, self).__init__()
        #         self.block = InvertedResidual(in_channels, out_channels, stride, expand_ratio=4)

        # class Net(nn.Module):
        #     def __init__(self, in_shape, out_shape, prm, device):
        #         super(Net, self).__init__()
        #         self.device = device
        #         in_channels = in_shape[1]
        #         num_classes = out_shape[0]
        #         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        #         self.bn1 = nn.BatchNorm2d(32)
        #         self.relu = nn.ReLU(inplace=True)
        #         self.layers = self.make_layers(in_channels, 32, prm, efficient_block=ExampleBlock)
        #         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        #         self.classifier = nn.Linear(32 * 6 * 6, num_classes)

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def make_layers(self, in_channels, out_channels, prm, efficient_block):
        layers = []
        repeat_times = 2
        for i in range(repeat_times):
            layers.append(
                efficient_block(in_channels if i == 0 else out_channels, out_channels, stride=2 if i == 0 else 1)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def __init_weight__(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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