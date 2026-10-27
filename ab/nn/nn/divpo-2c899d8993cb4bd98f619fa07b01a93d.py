import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
        num_classes = out_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Block(32, 32),
            Block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = nn.Sequential(
            Block(32, 64, stride=1, groups=32),
            Block(64, 64)
        )
        self.layer2 = nn.Sequential(
            Block(64, 128, stride=2, groups=64),
            Block(128, 128)
        )
        self.layer3 = nn.Sequential(
            Block(128, 128, stride=1, groups=128),
            Block(128, 128)
        )
        self.layer4 = nn.Sequential(
            Block(128, 256, stride=2, groups=128),
            Block(256, 256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
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

# Dummy input to determine model size (keep this as the minimum shape)
dummy_shape = [1, 3, 32, 32]
dummy_input = torch.randn(*dummy_shape)

# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Target parameter count
target_params = 500000

# Model configuration function
def configure_model(in_shape, out_shape, prm, device):
    net = Net(in_shape, out_shape, prm, device)
    current_params = count_parameters(net)

    # Tune the model size here
    while current_params > target_params:
        for child_name, child_module in net.named_children():
            if isinstance(child_module, Block):
                if child_name.endswith('_bn') or child_name.endswith('_act'):
                    continue  # Preserve batchnorm and activation layers

                for layer_name, layer in child_module.named_children():
                    if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                        if layer_name == 'conv':
                            # Remove padding from the first conv layer to reduce size
                            original_padding = layer.padding
                            layer.padding = 0
                            new_out_channels = layer.out_channels
                            layer.out_channels = layer.in_channels
                            layer.in_channels = new_out_channels
                            layer.kernel_size = (layer.kernel_size[0], layer.kernel_size[1])
                            layer.stride = (layer.stride[0], layer.stride[1])
                            layer._parameters['weight'] = layer._parameters['weight'].transpose(0, 1)
                            layer._parameters['weight'] = layer._parameters['weight'].contiguous()
                            layer._parameters['weight'] = layer._parameters['weight'].view(new_out_channels, layer.groups, layer.kernel_size[0], layer.kernel_size[1])
                            layer.groups = layer.groups * layer.in_channels
                            layer.padding = original_padding
                            break

                # Remove the entire block if the reduction in channels is not sufficient
                if child_module.conv.in_channels > child_module.conv.out_channels:
                    net._modules[child_name] = Block(child_module.conv.in_channels, child_module.conv.out_channels, child_module.conv.kernel_size, child_module.conv.stride, child_module.conv.padding, child_module.conv.groups)
                    break

        # Recalculate the parameter count
        current_params = count_parameters(net)

    # Ensure the final classifier has the correct number of output units
    in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]  # channels-first (C, H, W)
    num_classes = out_shape[0]
    net.classifier = nn.Linear(256 * 6 * 6, num_classes)

    # Final setup
    net.train_setup(prm)
    net.to(device)
    return net

# Entry point to define the model
def make_model(in_shape, out_shape, prm, device):
    return configure_model(in_shape, out_shape, prm, device)