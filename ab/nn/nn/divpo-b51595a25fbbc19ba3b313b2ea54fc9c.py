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
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False
        self._last_vars = {}

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )]
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def compile(self, prm):
        if not self._is_fwd_comp:
            self._last_vars = {}
            self._last_forwards = []
            for _ in range(prm.get('unrolled_iters', 1)):
                self._last_forwards.append([])
            self._is_fwd_comp = True
        self.unrolled_iters = prm.get('unrolled_iters', 1)
        self.unrolled_inner = prm.get('unrolled_inner', 1)
        self.unrolled_outer = prm.get('unrolled_outer', 1)
        self.unrolled_step = prm.get('unrolled_step', 0.1)

    def unroll_fwd(self, x, label):
        self._last_vars.clear()
        for _ in range(self.unrolled_iters):
            self._last_forwards.append([])
        x.requires_grad_(True)

        def closure():
            self.zero_grad()
            loss = self.learn([[x, label]])
            loss.backward()
            return loss

        for _ in range(self.unrolled_iters):
            self._last_vars.clear()
            for _ in range(self.unrolled_inner):
                x.grad = None
                loss = closure()
                for v in self.parameters():
                    if v.grad is None:
                        continue
                    if v not in self._last_vars:
                        self._last_vars[v] = v.grad.clone()
                    else:
                        self._last_vars[v].add_(v.grad)
                for v in self._last_vars:
                    v.grad = self._last_vars[v]
                self.step()
            for v in self._last_vars:
                v.grad = self._last_vars[v]
            self.step()
            self.zero_grad()
        return x

    def step(self):
        lr = self.unrolled_step
        for p in self.parameters():
            if p.grad is None:
                continue
            p.add_(p.grad, alpha=-lr)