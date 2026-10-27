import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = conv_bn(in_shape[1], 32, 3, 1, 1)
        self.conv2 = conv_bn(32, 64, 3, 2, 1)
        self.conv3 = conv_bn(64, 128, 3, 2, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._size = None

    def make_quant(self):
        self.conv1 = torch.quantization.quantize_dynamic(
            self.conv1, {torch.nn.Linear}, dtype=torch.qint8)
        self.conv2 = torch.quantization.quantize_dynamic(
            self.conv2, {torch.nn.Linear}, dtype=torch.qint8)
        self.conv3 = torch.quantization.quantize_dynamic(
            self.conv3, {torch.nn.Linear}, dtype=torch.qint8)
        self.avgpool = torch.quantization.quantize_dynamic(
            self.avgpool, {torch.nn.Linear}, dtype=torch.qint8)
        self.classifier = torch.quantization.quantize_dynamic(
            self.classifier, {torch.nn.Linear}, dtype=torch.qint8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _get_shape(self, _input):
        batch = _input.shape[0]
        return next(self.parameters()).shape

    def get_out_channels(self, which):
        if which == 'conv1': return self._get_shape(self.conv1(_input:=torch.zeros(1, *self.in_shape[1:])))[0]
        if which == 'conv2': return self._get_shape(self.conv2(_input:=torch.zeros(1, *self.in_shape[1:])))[0]
        if which == 'conv3': return self._get_shape(self.conv3(_input:=torch.zeros(1, *self.in_shape[1:])))[0]
        if which == 'pool': return self._get_shape(self.avgpool(_input:=torch.zeros(1, *self._get_shape(_input)[1:])))[1:]
        if which == 'linear': return self._get_shape(self.classifier(_input:=torch.zeros(1, *self._get_shape(_input)[1:])))[0]
        if which == 'pred': return self._get_shape(self.classifier(_input:=torch.zeros(1, *self._get_shape(_input)[1:])))[0]
        raise ValueError()

    def get_parameter_count(self, which):
        if which == 'conv1': return self.conv1.weight.shape.numel() + self.conv1.bias.shape.numel()
        if which == 'conv2': return self.conv2.weight.shape.numel() + self.conv2.bias.shape.numel()
        if which == 'conv3': return self.conv3.weight.shape.numel() + self.conv3.bias.shape.numel()
        if which == 'pool': return self.avgpool.weight.shape.numel() + self.avgpool.bias.shape.numel()
        if which == 'linear': return self.classifier[1].weight.shape.numel() + self.classifier[1].bias.shape.numel()
        if which == 'pred': return self.classifier[0].weight.shape.numel() + self.classifier[0].bias.shape.numel()
        raise ValueError()

    def get_out_size(self, which):
        if which == 'conv1': return self._get_shape(self.conv1(_input:=torch.zeros(1, *self.in_shape[1:])))[1:]
        if which == 'conv2': return self._get_shape(self.conv2(_input:=torch.zeros(1, *self.in_shape[1:])))[1:]
        if which == 'conv3': return self._get_shape(self.conv3(_input:=torch.zeros(1, *self.in_shape[1:])))[1:]
        if which == 'pool': return self._get_shape(self.avgpool(_input:=torch.zeros(1, *self._get_shape(_input)[1:])))[1:]
        if which == 'linear': return self._get_shape(self.classifier(_input:=torch.zeros(1, *self._get_shape(_input)[1:])))[1:]
        if which == 'pred': return self._get_shape(self.classifier(_input:=torch.zeros(1, *self._get_shape(_input)[1:])))[1:]
        raise ValueError()

    def get_weight(self, which):
        if which == 'conv1': return self.conv1.weight
        if which == 'conv2': return self.conv2.weight
        if which == 'conv3': return self.conv3.weight
        if which == 'pool': return self.avgpool.weight
        if which == 'linear': return self.classifier[1].weight
        if which == 'pred': return self.classifier[0].weight
        raise ValueError()

    def get_bias(self, which):
        if which == 'conv1': return self.conv1.bias
        if which == 'conv2': return self.conv2.bias
        if which == 'conv3': return self.conv3.bias
        if which == 'pool': return self.avgpool.bias
        if which == 'linear': return self.classifier[1].bias
        if which == 'pred': return self.classifier[0].bias
        raise ValueError()

    def get_dropout(self, which):
        if which == 'conv1': return self.conv1.dropout
        if which == 'conv2': return self.conv2.dropout
        if which == 'conv3': return self.conv3.dropout
        if which == 'pool': return self.avgpool.dropout
        if which == 'linear': return self.classifier.dropout
        if which == 'pred': return self.classifier.dropout
        raise ValueError()

    def get_nonlinearity(self, which):
        if which == 'conv1': return self.conv1.nonlinearity
        if which == 'conv2': return self.conv2.nonlinearity
        if which == 'conv3': return self.conv3.nonlinearity
        if which == 'pool': return self