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
        self.layer1 = self.build_layer(32, 32, 2, stride=1)
        self.layer2 = self.build_layer(32, 64, 2, stride=2)
        self.layer3 = self.build_layer(64, 128, 2, stride=2)
        self.layer4 = self.build_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_shape[0])
        )
        self._is_fwd_comp = False
        self._last_pruning_ratio = 0.0
        self._last_pruning_target = 0.0
        self.to(device)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm)

    
    def build_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)
    
    def build_optimizer(self, prm):
        return torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 5e-4)
        )

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
        self.optimizer = self.build_optimizer(prm)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def prune_by_channel(self, prune_ratio, target_ratio, prune_threshold):
        if self._is_fwd_comp:
            return
        if prune_ratio < self._last_pruning_ratio or prune_ratio > target_ratio:
            return
        self._last_pruning_ratio = prune_ratio
        self._last_pruning_target = target_ratio
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                prune_channels = int(m.out_channels * prune_ratio)
                weights = m.weight.data
                prune_mask = weights.abs().sum(dim=(1, 2, 3)) < prune_threshold
                prune_mask = prune_mask.unsqueeze(1).repeat(1, weights.size(1), 1, 1)
                m.weight.data = weights.masked_fill(prune_mask, 0)
                m.out_channels = m.out_channels - prune_channels
                m.bias.data = m.bias.data.masked_fill(prune_mask.sum(dim=(1, 2, 3)) > 0, 0)
                m.weight.data = m.weight.data / prune_ratio

    def prune_by_ratio(self, prune_ratio):
        if self._is_fwd_comp:
            return
        self._is_fwd_comp = True
        self._last_pruning_ratio = 0.0
        self._last_pruning_target = 0.0
        prune_threshold = 0.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                weights = m.weight.data
                weight_threshold = weights.abs().mean() * prune_threshold
                prune_mask = weights.abs() < weight_threshold
                prune_mask = prune_mask.unsqueeze(1).repeat(1, weights.size(1), 1, 1)
                m.weight.data = weights.masked_fill(prune_mask, 0)
                m.out_channels = m.out_channels - int(m.out_channels * prune_ratio)
                m.bias.data = m.bias.data.masked_fill(prune_mask.sum(dim=(1, 2, 3)) > 0, 0)
                m.weight.data = m.weight.data / (1 - prune_ratio)
        self._is_fwd_comp = False

    def prune_by_size(self, prune_ratio, target_ratio, prune_threshold):
        if self._is_fwd_comp:
            return
        if prune_ratio < self._last_pruning_ratio or prune_ratio > target_ratio:
            return
        self._last_pruning_ratio = prune_ratio
        self._last_pruning_target = target_ratio
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                weights = m.weight.data
                weight_threshold = weights.abs().mean() * prune_threshold
                prune_mask = weights.abs() < weight_threshold
                prune_mask = prune_mask.unsqueeze(1).repeat(1, weights.size(1), 1, 1)
                m.weight.data = weights.masked_fill(prune_mask, 0)
                m.out_channels = m.out_channels - int(m.out_channels * prune_ratio)
                m.bias.data = m.bias.data.masked_fill(prune_mask.sum(dim=(1, 2, 3)) > 0, 0)
                m.weight.data = m.weight.data / (1 - prune_ratio)

    def prune_by_threshold(self, prune_threshold):
        if self._is_fwd_comp:
            return
        self._is_fwd_comp = True
        self._last_pruning_ratio = 0.0
        self._last_pruning_target = 0.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                weights = m.weight.data
                weight_threshold = weights.abs