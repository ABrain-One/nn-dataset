import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 32, 2, 1)
        self.layer2 = self._make_layer(32, 64, 2, 2)
        self.layer3 = self._make_layer(64, 128, 2, 2)
        self.layer4 = self._make_layer(128, 256, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, out_shape[0])

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True)]
        for i in range(1, blocks):
            layers += [nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(out_planes),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

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

def check_shape(shape):
    assert isinstance(shape, (list, tuple))
    assert len(shape) == 4
    assert shape[1] in {1, 3}
    assert all(isinstance(s, int) and s > 0 for s in shape[2:])
    assert shape[0] in {1, 32, 64}  # batch size

def get_eps(shape):
    check_shape(shape)
    return 1e-3 + 0.99 * (1 - 1 / (shape[0] * shape[2] * shape[3]))

def get_rng():
    return torch.manual_seed(42)

def get_data_gen(shape, prm):
    eps = get_eps(shape)
    rng = get_rng()
    return generate(shape[1:], eps, rng, prm)

def generate(size, eps, rng, prm):
    def get_img():
        for _ in range(10):
            img = torch.empty(size, dtype=torch.float32, device='cpu').uniform_(-eps, eps)
            if len(size) == 3 and img.sum() > 0.01:  # color
                return img
            if len(size) == 2 and img.sum() < 0.05:  # gray
                return img
        return torch.zeros(size, dtype=torch.float32, device='cpu')
    return (get_img() for _ in range(prm['batch_size'] // prm['world_size']))

def train(net, prm):
    train_data = get_data_gen(net.get_in_shape(), prm)
    net.train_setup(prm)
    net.learn(train_data)

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_in_shape(net):
    return net.get_in_shape()

def trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def get_class():
    return Net