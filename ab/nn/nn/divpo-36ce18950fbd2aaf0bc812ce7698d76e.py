import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W) — derive from in_shape
        num_classes = out_shape[0]  # number of target classes
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.make_layer(32, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def make_layer(self, inplanes, planes, blocks):
        layers = [nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        for _ in range(1, blocks):
            layers += [nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.02),
            momentum=prm.get('momentum', 0.9),
            weight_decay=0.0001
        )
    class WeightedRandomSampler(torch.utils.data.sampler.Sampler):
        """Samples elements randomly from a given list of indices with replacement.
        If the list of indices is a numpy.array, it will be converted into a
        Python list.
        The probability of picking an index is proportional to the provided weights.
        Arguments:
            weights (sequence)   : weights for sampling
            num_samples (int)    : number of samples to draw
        """

        def __init__(self, weights, num_samples=None):
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            self.num_samples = num_samples

            if self.weights.dim() == 1:
                self.weights = self.weights.view(-1)
            else:
                raise ValueError("weights should be 1D: target weights are {}, "
                                 "but got weights of size {}".format(self.weights.size(), self.weights.dim()))

            if not self.weights.dtype == torch.double:
                self.weights = self.weights.double()

            self.cumulative_weights = self.weights.cumsum(0)
            self.total_weight = self.cumulative_weights[-1]

        def __len__(self):
            return self.num_samples

        def __iter__(self):
            if self.num_samples is None:
                return iter(torch.multinomial(self.weights, len(self.weights)).tolist())
            else:
                return iter(torch.multinomial(self.weights, self.num_samples).tolist())