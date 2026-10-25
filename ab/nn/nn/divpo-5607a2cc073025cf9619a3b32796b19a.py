import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Add this class here (don't inherit from Net): it should include the
# resnet50 block implementation (without docstring). You can find similar
# blocks in submitted models.
# For example, blocks for resnet50:
# https://github.com/huaweicloud/Swin-Transformer/blob/4d718e9/models/swin_transformer.py#L38
# Or those for resnext:
# https://github.com/pytorch/vision/blob/4a1c71b/torchvision/models/resnext.py#L20
class Resnet50Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=1e-3, affine=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=1e-3),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=1e-3, affine=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.resblock(x)
        residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


# Here is the Net class to be completed.
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.learning_rate = prm.get('lr', 0.001)
        self.momentum = prm.get('momentum', 0.9)

        self.build_model(out_shape)

    def build_model(self, out_shape: tuple) -> None:
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            Resnet50Block(in_channels=64, out_channels=128, stride=1),
            Resnet50Block(in_channels=128, out_channels=256, stride=2),
            Resnet50Block(in_channels=256, out_channels=512, stride=2),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),

            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, int(out_shape[0]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
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
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()