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
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, out_shape[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.to(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0005
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.optimizer.lr = prm['lr']
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

    def get_complexity(self, input_resolution):
        # Estimate MACs and parameters for complexity analysis.
        # This is a simple approximation.
        macs, params = 0, 0
        x = torch.randn(1, *input_resolution, device=self.device)
        for module in self.children():
            macs, params = profile(module, inputs=(x,), custom_ops={ReNorm: custom_op}, verbose=False)
        return macs, params

def custom_op(gm, *args):
    # Replacing ReNorm with a simplified version.
    # This is a placeholder for a more comprehensive analysis.
    for node in gm.nodes:
        if node.op == 'prim::CallMethod':
            if node.target == 'norm':
                norm_type = node.kwargs.get('norm_type', 2)
                if norm_type == 2:
                    # Replacing ReNorm with standard BatchNorm2d
                    new_module = nn.BatchNorm2d(node.args[0].size(-1))
                    new_module.weight.data.copy_(node.kwargs['weight'])
                    new_module.bias.data.copy_(node.kwargs['bias'])
                    new_module.running_mean.data.copy_(node.kwargs['running_mean'])
                    new_module.running_var.data.copy_(node.kwargs['running_var'])
                    new_module.eps = node.kwargs['eps']
                    return new_module

class ReNorm(nn.Module):
    # Simplified ReNorm for custom_op
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        norm = torch.norm(x.reshape(x.shape[0], -1), dim=1, p=2, keepdim=True)
        x = x / (norm + self.eps)
        x = x * self.weight.view(1, -1) + self.bias.view(1, -1)
        return x

def profile(module, inputs, custom_ops, verbose):
    # Placeholder for torch.profiler.profile to estimate MACs and parameters.
    # This is a simplified version that does not accurately measure complexity.
    if verbose:
        print(f"Analyzing {module}")
    # Estimate MACs (Multiply-Accumulate operations)
    macs = 0
    for name, child_module in module.named_children():
        if name == 'stem':
            # Example: Conv2d with 3 input channels and 32 output channels
            child_inputs = inputs[0]
            _, child_macs = torch.profile(child_module, inputs=(child_inputs,), record_shapes=True)
            macs += child_macs
        elif name == 'layer1' or name == 'layer2':
            # Example: Sequential with Conv2d and MaxPool2d
            child_inputs = inputs[0]
            _, child_macs = torch.profile(child_module, inputs=(child_inputs,), record_shapes=True)
            macs += child_macs
        elif name == 'avgpool':
            # Example: AdaptiveAvgPool2d with output size (6, 6)
            child_inputs = inputs[0]
            _, child_macs = torch.profile(child_module, inputs=(child_inputs,), record_shapes=True)
            macs += child_macs
        elif name == 'classifier':
            # Example: Linear with input features 128 * 6 * 6 and output features 10
            child_inputs = inputs[0].view(inputs[0].size(0), -1)
            child_module = module[name]
            _, child_macs = torch.profile(child_module, inputs=(child_inputs,), record_shapes=True)
            macs += child_macs
    # Estimate parameters
    params = sum(p.numel() for p in module.parameters())
    return macs, params