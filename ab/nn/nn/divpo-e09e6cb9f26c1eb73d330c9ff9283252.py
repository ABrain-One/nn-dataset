import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def get_conv_output_shape(in_shape, conv_params):
    batch_size = in_shape[0]
    channels = in_shape[1]
    height = in_shape[2]
    width = in_shape[3]

    conv = nn.Conv2d(channels, *conv_params)
    out_channels = conv.out_channels

    # Apply the conv layer to a dummy input tensor
    dummy_input = torch.randn(batch_size, channels, height, width)
    output = conv(dummy_input)

    # Calculate the output height and width
    output_height = output.shape[2]
    output_width = output.shape[3]

    return (out_channels, output_height, output_width)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        channels = in_shape[1]
        height = in_shape[2]
        width = in_shape[3]
        num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Calculate the output shape of the first convolution
        conv1_params = (32, 3, 3, 1, 1)
        conv1_out_shape = get_conv_output_shape(in_shape, conv1_params)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the rest of the layers based on the output shape
        layer_params = [
            (32, 64, 3, 1, 1),
            (64, 128, 3, 1, 1),
            (128, 256, 3, 1, 1),
            (256, 256, 3, 1, 1),
            (256, 512, 3, 1, 1)
        ]
        layer_blocks = []
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(layer_params):
            layer_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            layer_blocks.append(nn.BatchNorm2d(out_channels))
            layer_blocks.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layer_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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