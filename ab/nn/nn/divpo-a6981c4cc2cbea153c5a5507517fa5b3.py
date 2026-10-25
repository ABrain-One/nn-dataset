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
        self.block1 = self._make_block(32, 64, 2, prm.get('dropout', 0.0))
        self.block2 = self._make_block(64, 128, 2, prm.get('dropout', 0.0))
        self.block3 = self._make_block(128, 256, 2, prm.get('dropout', 0.0))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )
        self._size_limit = 500000
        self.prune_target_macs = None
        self.apply(self._weight_init)

    def _make_block(self, in_channels, out_channels, layers, dropout):
        layers_list = []
        for i in range(layers):
            layers_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers_list.append(nn.BatchNorm2d(out_channels))
            layers_list.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers_list)

    def _weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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

    def prune(self):
        if self.prune_target_macs is not None:
            rmb_before = self._get_model_size()
            pruned = self._prune_to_size(self.prune_target_macs)
            rmb_after = self._get_model_size()
            pruned_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            if rmb_after + pruned_params > self._size_limit:
                self.prune_target_macs = None
                return False
            print(f'Pruned {pruned} ({100. * pruned / rmb_before:.2f}%) to {rmb_after} ({100. * rmb_after / self._size_limit:.2f}%)')
            return True
        return False

    def _get_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _prune_to_size(self, target_macs):
        module_types = (nn.Conv2d, nn.Linear)
        pruned = 0
        for module in self.modules():
            if isinstance(module, module_types):
                if module.weight.requires_grad:
                    original_macs = module.weight.numel() * module.in_channels * module.out_channels
                    if original_macs > target_macs:
                        pruning_ratio = target_macs / original_macs
                        pruned_filters = int(module.out_channels * pruning_ratio)
                        pruned_channels = int(module.in_channels * pruning_ratio)
                        module.in_channels = pruned_channels
                        module.out_channels = pruned_filters
                        module.weight.data = module.weight.data[:, :pruned_filters, :pruned_channels]
                        pruned += original_macs - (module.weight.numel() * module.in_channels * module.out_channels)
        return pruned

    def named_prunable_modules(self):
        module_types = (nn.Conv2d, nn.Linear)
        prunable_modules = []
        for name, module in self.named_modules():
            if isinstance(module, module_types) and module.weight.requires_grad:
                prunable_modules.append((name, module))
        return prunable_modules