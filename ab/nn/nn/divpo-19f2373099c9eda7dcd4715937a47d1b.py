import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.layer0 = conv_bn(self.in_channels, 32, 2)
        self.layer1 = conv_dw(32, 32, 1)
        self.layer2 = conv_dw(32, 64, 2)
        self.layer3 = conv_dw(64, 64, 1)
        self.layer4 = conv_dw(64, 128, 2)
        self.layer5 = conv_dw(128, 128, 1)
        self.layer6 = conv_dw(128, 128, 1)
        self.layer7 = conv_dw(128, 128, 1)
        self.layer8 = conv_dw(128, 128, 1)
        self.layer9 = conv_dw(128, 128, 1)
        self.layer10 = conv_dw(128, 128, 1)
        self.layer11 = conv_dw(128, 128, 1)
        self.layer12 = conv_dw(128, 128, 1)
        self.layer13 = conv_dw(128, 128, 1)
        self.layer14 = conv_dw(128, 128, 1)
        self.layer15 = conv_dw(128, 128, 1)
        self.layer16 = conv_dw(128, 128, 1)
        self.layer17 = conv_dw(128, 128, 1)
        self.layer18 = conv_dw(128, 128, 1)
        self.layer19 = conv_dw(128, 128, 1)
        self.layer20 = conv_dw(128, 128, 1)
        self.layer21 = conv_dw(128, 128, 1)
        self.layer22 = conv_dw(128, 128, 1)
        self.layer23 = conv_dw(128, 128, 1)
        self.layer24 = conv_dw(128, 128, 1)
        self.layer25 = conv_dw(128, 128, 1)
        self.layer26 = conv_dw(128, 128, 1)
        self.layer27 = conv_dw(128, 128, 1)
        self.layer28 = conv_dw(128, 128, 1)
        self.layer29 = conv_dw(128, 128, 1)
        self.layer30 = conv_dw(128, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.layer26(x)
        x = self.layer27(x)
        x = self.layer28(x)
        x = self.layer29(x)
        x = self.layer30(x)
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