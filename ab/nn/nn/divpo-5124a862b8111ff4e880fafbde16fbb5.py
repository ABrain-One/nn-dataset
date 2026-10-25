import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )

        for _ in range(2):
            self.add_module('res' + str(_), self.resblock(128))

        self._init_weight()


    def resblock(self, channels):
        layers = []
        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        ]
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def check_shape(shape, expected_shape):
    if shape != expected_shape:
        raise ValueError(f"Unexpected shape {shape}, expected {expected_shape}")

def get_workspace_size(dtype):
    return torch.empty(1, dtype=dtype).element_size()

def get_workspace(size, dtype):
    return torch.empty(size, dtype=dtype)

class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = int(inp * expand_ratio)
        self.conv1 = torch.nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(hidden_dim)
        self.act1 = torch.nn.ReLU6(inplace=True)

        self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(hidden_dim)
        self.act2 = torch.nn.ReLU6(inplace=True)

        self.conv3 = torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(oup)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.act2(self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x))))))
        else:
            return self.act2(self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x))))))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv2, self.bn2)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1, self.bn1)
        kernelid, biasid = self._fuse_bn_tensor(self.conv3, self.bn3)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            pad_before = [(i // 2, i // 2 + i % 2) for i in kernel1x1.shape[-2:]]
            pad_before = [item for sublist in pad_before for item in sublist]
            kernel = torch.nn.functional.pad(kernel1x1, [0, 0] + pad_before + [0, 0])
            return kernel

    def _fuse_bn_tensor(self, conv, bn):
        if not self.training:
            conv_w = conv.weight.data
            conv_b = (conv.bias or torch.zeros(conv_w.size(0))).data
            bn_rm = bn.running_mean
            bn_riv = bn.running_var
            bn_r = bn.weight.data
            bn_b = bn.bias.data
            if conv.bias is None:
                bn_rm = bn_rm.unsqueeze(-1)
                bn_riv = bn_riv.unsqueeze(-1)
                conv_b = conv_b.unsqueeze(-1)
            eps = bn.eps
            sqrt_bn_riv