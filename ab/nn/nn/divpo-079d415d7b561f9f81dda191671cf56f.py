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
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])
        self.to(device)
        self.criteria = nn.CrossEntropyLoss()
        lr, momentum = prm.get('lr', 1e-3), prm.get('momentum', 0.9)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def get_flops(self, input_size):
        flops = 0
        x = torch.rand(size=input_size).to(self.device)
        flops_counter = torch.profiler.flops_counter_utils.add_flops_counting_methods(self)
        with torch.no_grad():
            flops_counter(x)
            flops += flops_counter.total()
        return flops

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_complexity(self, input_size):
        flops = self.get_flops(input_size)
        params = self.count_parameters()
        # Assume 2.5GHz CPU, 1.0E-9 FLOPS/byte, 1.0E-6 MACs/FLOP, 2.5E-8 MACs/byte
        # FLOPS ~= MACs * 2 * 1.0E-9, MACs ~= FLOPS / 2 * 1.0E9
        # BYTE_COUNT = FLOPS / FLOP_PER_BYTE
        # MACs_PER_BYTE = BYTE_COUNT * MACs_PER_BYTE
        # MACs_PER_BYTE ~= FLOPS / MAC_PER_FLOP * MACs_PER_BYTE
        # BYTE_COUNT ~= FLOPS / FLOP_PER_BYTE / MAC_PER_BYTE
        # BYTE_COUNT ~= FLOPS / FLOP_PER_BYTE * MACs_PER_BYTE / MAC_PER_BYTE
        # BYTE_COUNT ~= FLOPS / FLOP_PER_BYTE * MACs_PER_BYTE
        byte_count = flops / 2.5E-9 * (1.0E-9 * 2.5E-8)
        return {'params': params, 'flops': flops, 'byte_count': byte_count}