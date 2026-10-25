import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

    def _make_layer(self, inplanes, planes, blocks):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_breakdown(self):
        # Breakdown memory per layer
        def get_layer_size(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        breakdown = {
            str(k): get_layer_size(v) for k, v in self.named_children()
            if not isinstance(v, nn.Sequential) or len(list(v.children())) > 1
        }
        total = sum(breakdown.values())
        breakdown['total'] = total
        return breakdown

    def get_memory(self, image_size):
        # Estimate memory usage for a single image
        batch = torch.zeros((1, *image_size), device=self.device)
        batch_size = next(self.parameters()).device.numel()
        self.forward(batch)
        return batch_size * image_size[1] * image_size[2] * 4 / 1024 ** 2