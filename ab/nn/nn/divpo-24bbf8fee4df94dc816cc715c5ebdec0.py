import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_resblock(128, 128)
        self.layer5 = self.build_resblock(128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self._is_fwd_complied = False
        self.to(device)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.01)
        self.criteria = (nn.CrossEntropyLoss(),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride if i == 0 else 1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def build_resblock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def build_optimizer(self, prm, parameters):
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.01)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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

def check_shape(shape):
    p = shape.index('p')
    return (shape[:p] + shape[p+1:])

def export_model(config):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: str) -> None:
            super().__init__()
            self.model = build_model(in_shape, out_shape, prm, device)

        def forward(self, x):
            return self.model(x)

    def build_model(in_shape: tuple, out_shape: tuple, prm: dict, device: str) -> nn.Module:
        mdl_cls = AutoModelForCausalLM
        in_shape = check_shape(in_shape)
        model = mdl_cls(config=AutoConfig.from_pretrained("gpt2"))
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)
        return model

    model_class = Net
    return model_class