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
        self.layer1 = self.build_layer(32, 32, 2, first=True)
        self.layer2 = self.build_layer(32, 64, 2)
        self.layer3 = self.build_layer(64, 128, 2)
        self.layer4 = self.build_resblock(128, 128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False

    def build_layer(self, in_channels, out_channels, stride, first=False):
        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def build_resblock(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(self.build_layer(in_channels, out_channels, stride=1 if in_channels == out_channels else 2))
        for _ in range(repeat - 1):
            layers.append(self.build_layer(out_channels, out_channels, stride=1))
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

    def compile(self):
        if not self._is_fwd_comp:
            self.cpu()
            self.eval()
            self._is_fwd_comp = True
            self.forward = self._fast_forward

    def _fast_forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def _slow_forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    def get_penultimate(self):
        return nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )

    def get_loss(self, x, label):
        return self.forward(x).losses

    def get_optimizer(self, prm):
        return torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def get_lr(self, prm):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return prm.get('lr', 0.01)

    def get_model_size(self):
        return sum(p.numel() for p in self.parameters())

    def get_flops(self):
        x = torch.rand(1, *self.stem[0].in_shape, device=self.device)
        x = self.stem(x)
        x = x.mean(dim=(2, 3))
        return x.numel()

    def get_latency(self):
        x = torch.rand(1, *self.stem[0].in_shape, device=self.device)
        return self.profile(x, 1)

    def profile(self, x, repeat):
        self.eval()
        x = x.to(self.device)
        self.to('cpu')
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            x = self.forward(x)
            x = x.mean()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    def get_device(self):
        return self.device

    def get_param_groups(self):
        return self.param_groups

    def get_param_groups_opt(self):
        return self.optimizer.param_groups

    def get_param_groups_diff(self, prm):
        groups = self.get_param_groups_opt()
        lr = prm.get('lr', 0.01)
        momentum = prm.get('momentum', 0.9)
        return [[{'lr': lr, 'momentum': momentum},
                 {'params': g['params']}] for g in groups]

    def get_param_groups_opt_diff(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_opt_diff_opt(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_diff_opt(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_diff_opt_diff(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_diff_opt_diff_opt(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_diff_opt_diff_opt_diff(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_diff_opt_diff_opt_diff_diff(self, prm):
        return self.get_param_groups_diff(prm)

    def get_param_groups_diff_opt_diff_opt_diff_diff_diff(self, prm):
        return self.get_param_groups_