import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Use this type of block (with self.transform and self.act):
class Layer(nn.Module):
    def __init__(self, in_chs, out_chs, act, stride):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            act
        )
        self.act = act

    def forward(self, x):
        return self.transform(x)

    def fuse(self):
        nn.intrinsic.pointwise_conv_2d.PointwiseConv2d.fuse_pointwise_conv_batchnorm(
            self.transform[0], self.transform[1], self.transform[2]
        )


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # derived from in_shape: in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.model = self.build_model()

    def build_model(self):
        # Block used in the model for the first time (you need to derive from Layer)
        block_class = Layer
        layers = [
            block_class(self.in_channels, 32, nn.ReLU(inplace=True), 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            block_class(32, 64, nn.ReLU(inplace=True), 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            block_class(64, 128, nn.ReLU(inplace=True), 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(6),
        ]
        self.conv1 = nn.Sequential(*layers)
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)
        self._reset_parameters()
        return nn.Sequential(*layers)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
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
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()