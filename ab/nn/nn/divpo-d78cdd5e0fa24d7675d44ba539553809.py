import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(32, 32, 3, 1, 1)
        self.layer2 = self.make_layer(32, 64, 3, 2, 1)
        self.layer3 = self.make_layer(64, 128, 3, 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)

        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def make_layer(self, in_channels, out_channels, block_num, stride, padding):
        layers = []
        layers.append(Block(in_channels, out_channels, 3, stride, 1))
        for _ in range(1, block_num):
            layers.append(Block(out_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            self.optimizer.step()

# Generate a random 500000-parameter model
# This is a dummy value, as the actual parameter count will be determined by the model structure and choices made during the design process.
# The actual number of parameters should be less than or equal to 500000.
random_parameter_count = 500000

# This is a placeholder for the actual design process.
# The following functions should be implemented to generate a valid model.
# - choose_kernel_size
# - choose_padding
# - choose_dilation
# - choose_groups
# - choose_block_config
# - choose_block_type
# - choose_block_channels
# - choose_block_num
# - choose_block_stride
# - choose_block_bias
# - choose_first_conv_channels
# - choose_layer_channels
# - choose_layer_num
# - choose_layer_stride
# - choose_layer_padding
# - choose_layer_dilation
# - choose_layer_groups
# - choose_layer_bias
# - choose_layer_block_type
# - choose_layer_block_channels
# - choose_layer_block_num
# - choose_layer_block_stride
# - choose_layer_block_padding
# - choose_layer_block_dilation
# - choose_layer_block_groups
# - choose_layer_block_bias
# - choose_layer_block_block_type
# - choose_layer_block_block_channels
# - choose_layer_block_block_num
# - choose_layer_block_block_stride
# - choose_layer_block_block_padding
# - choose_layer_block_block_dilation
# - choose_layer_block_block_groups
# - choose_layer_block_block_bias
# - choose_classifier_units
# - choose_classifier_dropout
# - choose_classifier_activation
# - choose_pooling
# - choose_regularization
# - choose_first_conv_stride
# - choose_first_conv_padding
# - choose_first_conv_dilation
# - choose_first_conv_groups
# - choose_first_conv_bias
# - choose_first_conv_block_type
# - choose_first_conv_block_channels
# - choose_first_conv_block_num
# - choose_first_conv_block_stride
# - choose_first_conv_block_padding
# - choose_first_conv_block_dilation
# - choose_first_conv_block_groups
# - choose_first_conv_block_bias
# - choose_first_conv_block_block_type
# - choose_first_conv_block_block_channels
# - choose_first_conv_block_block_num
# - choose_first_conv_block_block_stride
# - choose_first_conv_block_block_padding
# - choose_first_conv_block_block_dilation
# - choose_first_conv_block_block_groups
# - choose_first_conv_block_block_bias
# - choose_first_conv_block_block_block_type
# - choose_first_conv_block_block_block_channels
# - choose_first_conv_block_block_block_num
# - choose_first_conv_block_block_block_stride
# - choose_first_conv_block_block_block_padding
# - choose_first_conv_block_block_block_dilation
# - choose_first_conv_block_block_block_groups
# - choose_first_conv_block_block_block_bias
# - choose_first_conv_block_block_block_block_type
# - choose_first_conv_block_block_block_block_channels
# - choose_first_conv_block_block_block_block_num
# - choose_first_conv_block_block_block_block_stride
# - choose_first_conv_block_block_block_block_padding
# - choose_first_conv_block_block_block_block_dilation
# - choose_first_conv_