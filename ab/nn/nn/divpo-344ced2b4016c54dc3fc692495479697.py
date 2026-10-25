import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0005
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def get_complexity(self, x: torch.Tensor) -> int:
        total_params = sum(p.numel() for p in self.parameters())
        input_size = x.shape[1]
        kernel_ops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kernel_ops += m.weight.shape[2] * m.weight.shape[3]
            if isinstance(m, nn.Linear):
                kernel_ops += m.weight.shape[1]
        total_ops = total_params * input_size * kernel_ops
        return total_ops

    def get_memory_budget(self, shape: tuple) -> int:
        input_size = torch.empty(*shape).numel()
        total_memory = input_size * 4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                total_memory += m.weight.numel() * 4
            if isinstance(m, nn.Linear):
                total_memory += m.weight.numel() * 4
        return total_memory

    def get_flops_budget(self, x: torch.Tensor) -> int:
        input_size = x.shape[1]
        total_flops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                total_flops += m.weight.shape[2] * m.weight.shape[3]
            if isinstance(m, nn.Linear):
                total_flops += m.weight.shape[1]
        return total_flops