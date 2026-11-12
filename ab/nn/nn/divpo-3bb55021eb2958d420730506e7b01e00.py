import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.num_classes = out_shape[0]
        self.features = nn.Sequential(
            # 3x3 conv
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # inverted residual blocks
            InvertedResidual(32, 32, stride=1, expand_ratio=3),
            InvertedResidual(32, 64, stride=2, expand_ratio=3),
            InvertedResidual(64, 128, stride=2, expand_ratio=3),
            InvertedResidual(128, 256, stride=2, expand_ratio=3),
            InvertedResidual(256, 512, stride=2, expand_ratio=3),

            # output
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1280, kernel_size=1, stride=1, bias=False),
        )

        self.classifier = nn.Linear(1280, self.num_classes)
        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

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
class Net(MobileNetV3):
    """MobileNetV3-like backbone + head (1280-dimensional vectors)."""
    def __init__(self, in_shape, out_shape, prm, device):
        emb_dim = 1280
        super().__init__(in_shape, out_shape, prm, device)
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(emb_dim, out_shape[0]),
        )

    def learn(self, train_data):
        self.train()
        for i, (x, y) in enumerate(train_data):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            out = self(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            if i % 5 == 0:
                self.head[-1].weight.data *= 0.99
### Response: