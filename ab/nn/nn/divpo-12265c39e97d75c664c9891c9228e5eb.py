import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        oup = oup // 4

        self.conv = nn.Sequential(
            # t channel
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),

            # k3 s2 p1
            nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),

            # t channel
            nn.Conv2d(oup, oup * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup * 4),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 3),
            InvertedResidual(32, 32, 1, 4),
            InvertedResidual(32, 32, 1, 5),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._is_tracing = False

    def add_weight(self, name, w):
        if not self._is_tracing:
            w = w.to(self.device)
        self.register_parameter(name, nn.Parameter(w))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria[0](output, target)
            loss.backward()
            self.optimizer.step()
### Response: