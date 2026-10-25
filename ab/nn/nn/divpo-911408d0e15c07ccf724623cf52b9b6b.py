import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# This is a template for a new model.
# Add the methods of the model here. The name of the class should be Net.


# Use this class to share your experimental blocks (like Block3, MixLayer, etc).
class BlockTemplate(nn.Module):
    # BlockTemplate must implement __init__ and forward methods.
    # Use these for your experimental blocks, they will be used in tests.
    def __init__(self):
        super(BlockTemplate, self).__init__()
        self.relprop_weight = 0.1

    def forward(self, x):
        return x

    def relprop(self, grad_output, *args, **kwargs):
        return self.relprop_weight * grad_output


# This is the Net class that will be used for training.
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self._last_channels, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # BlockTemplate is a placeholder for your experimental blocks
        # Replace it with your Block3 or MixLayer or other block you're experimenting with
        layers += [BlockTemplate() for _ in range(2)]  # repeat as needed

        layers += [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        self._last_channels = 64
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
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
            self.optimizer.step()