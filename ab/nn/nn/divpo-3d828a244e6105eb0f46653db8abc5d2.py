import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Define Net class with the following layout:
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # this will be 3 for RGB
        self.num_classes = out_shape[0]

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self.make_layer(32, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def make_layer(self, in_channels, out_channels, blocks):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(1, blocks):
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 5e-4)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()


# Expected architecture (input block to replicate):
# Net with patterns:
#   - Transpose(2, 1) before self-attention (encoder block)
#   - layer_norm[i] after each FFN block (decoder block)
#   - edge-friendly activations (e.g., leaky_relu or hardtanh): leaky_relu(self.ffn[i](x))