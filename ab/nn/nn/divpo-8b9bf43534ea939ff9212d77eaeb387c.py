import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup)
    )

class YoloDetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(YoloDetectionHead, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        out = self.conv(x)
        batch_size, _, grid_h, grid_w = out.shape
        out = out.view(batch_size, self.num_anchors, self.num_classes + 5, grid_h, grid_w)
        return out

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.features = self.build_features()
        self.yolo_head = YoloDetectionHead(128, 9, self.num_classes)
        self.classifier = nn.Linear(128, self.num_classes)

        self._initialize_weights()

    def build_features(self):
        layers = []
        layers += [
            conv_bn(self.in_channels, 32, stride=2, leaky=0.1),
            conv_bn(32, 64, stride=2, leaky=0.1),
            conv_bn(64, 128, stride=2, leaky=0.1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3), keepdim=True)
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}

def supported_hyperparameters():
    return {'lr', 'momentum'}