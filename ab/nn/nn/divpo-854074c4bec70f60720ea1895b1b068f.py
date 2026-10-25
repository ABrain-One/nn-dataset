import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# Optional: helper blocks BEFORE Net
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            # Pointwise
            nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(32, 32, 1, 2),
            InvertedResidual(32, 32, 1, 2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self._set_state_dict_type()

    def _set_state_dict_type(self):
        if 'get_extra_state' in nn.CrossEntropyLoss().__dict__:
            # extra_state_type will be one of: "NotRequired", "Optional[Any]", "Dict[str, Any]"
            # We will use "Optional[Any]" to be future-proof (PyTorch 2.0)
            self.extra_state_type = nn.CrossEntropyLoss().get_extra_state()
        else:
            self.extra_state_type = None

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            if self.extra_state_type is not None:
                loss.backward(create_graph=True)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()