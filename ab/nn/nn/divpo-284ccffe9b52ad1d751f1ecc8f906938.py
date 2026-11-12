import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = int(out_shape[0])
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.block7 = Block(64, 64)
        self.block8 = Block(64, 64)
        self.block9 = Block(64, 64)
        self.block10 = Block(64, 64)
        self.block11 = Block(64, 64)
        self.block12 = Block(64, 64)
        self.block13 = Block(64, 64)
        self.block14 = Block(64, 64)
        self.block15 = Block(64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.pool(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}


######################
# Early-stop helper
# (may be unused for best edge-perf; see E10-2-E10-2)
######################
class E10:
    def __init__(self, m, p, E10_2=False):
        self.m = m; self.p = p; self.E10_2 = E10_2
        if E10_2: self.E10_2_p = E10(self.m, self.p.m2)

    def m1(self):
        return E10(self.m, self.p, E10_2=True)

    def m2(self):
        if not self.E10_2: raise Exception('no m2')
        return E10(self.m, self.E10_2_p, E10_2=False)

    def setup(self):
        m1 = self.m1(); m2 = self.m2()
        m1.m.p = m2.m.p; m1.p = self.p; m2.m = m1
        if self.E10_2: m1.E10_2_p = m2.m.m2; m2.m.m2 = m1
        self.m = m1; self.m.m2 = m2

    def _get_layer(self):
        if self.E10_2:
            E10_2 = self.m.m2._get_layer()
            return self.m.m2.m._get_layer() + E10_2
        else:
            return self.m._get_layer()

    def _get_param(self):
        if self.E10_2:
            E10_2_p = self.m.m2._get_param()
            return self.m.m2.m._get_param() + E10_2_p
        else:
            return self.m._get_param()

    def _get_weight(self):
        if self.E10_2:
            E10_2_w = self.m.m2._get_weight()
            return self.m.m2.m._get_weight() + E10_2_w
        else:
            return self.m._get_weight()

    def _get_bias(self):
        if self.E10_2:
            E10_2_b = self.m.m2._get_bias()
            return self.m.m2.m._get_bias() + E10_2_b
        else:
            return self.m._get_bias()

    def _get_bias_in(self):
        if self.E10_2:
            E10_2_b = self.m.m2._get_bias_in()
            return self.m.m2.m._get_bias_in() + E10_2_b
        else:
            return self.m._get_bias_in()

    def _get_bias_out(self):
        if self.E10_2:
            E10_2_b = self.m.m2._get_bias_out()
            return self.m.m2.m._get_bias_out() + E10_2_b
        else:
            return self.m._get_bias_out()

    def _get_dropout(self):
        if self.E10_2:
            E10_2_d = self.m.m2._get_dropout()
            return self.m.m2.m._get_dropout() + E10_2_d
        else:
            return self.m._get_dropout()

    def _get_bias_bias(self):
        if self.E10_2:
            E10_2_b = self.m.m2._get_bias_bias()
            return self.m.m2.m._get_bias_bias() + E10_2_b
        else:
            return self.m._get_bias_