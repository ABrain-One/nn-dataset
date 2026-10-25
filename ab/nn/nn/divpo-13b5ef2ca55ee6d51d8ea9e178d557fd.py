import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(leaky, inplace=True)
    )

def conv_dw(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(leaky, inplace=True),
    )

class YoloLoss(nn.Module):
    def __init__(self, num_classes):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        # Extract dimensions
        batch_size = pred.shape[0]
        num_boxes = pred.shape[1] // (5 + self.num_classes)
        width = pred.shape[2]
        height = pred.shape[3]

        # Extract anchors
        anchor_w = self.anchors[:, 0]
        anchor_h = self.anchors[:, 1]

        # Extract predictions
        pred_xy = pred[..., :2]
        pred_wh = pred[..., 2:4]
        pred_confidence = pred[..., 4:5]
        pred_classes = pred[..., 5:]

        # Extract targets
        target_xy = target[..., :2]
        target_wh = target[..., 2:4]
        target_confidence = target[..., 4:5]
        target_classes = target[..., 5:]

        # Calculate loss for xy
        loss_xy = F.mse_loss(pred_xy, target_xy)

        # Calculate loss for wh
        loss_wh = F.mse_loss(pred_wh, target_wh)

        # Calculate loss for confidence
        loss_confidence = F.binary_cross_entropy_with_logits(pred_confidence, target_confidence)

        # Calculate loss for classes
        loss_classes = F.binary_cross_entropy_with_logits(pred_classes, target_classes)

        # Calculate total loss
        total_loss = loss_xy + loss_wh + loss_confidence + loss_classes

        return total_loss

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])
        self.yolo_loss = YoloLoss(self.num_classes)
        self._create_layers()

    def _create_layers(self):
        self.features = nn.Sequential(
            conv_bn(self.in_channels, 32, stride=2, leaky=0.1),
            conv_dw(32, 64, stride=2, leaky=0.1),
            conv_dw(64, 128, stride=2, leaky=0.1),
            conv_dw(128, 256, stride=2, leaky=0.1),
            conv_dw(256, 512, stride=2, leaky=0.1),
            conv_dw(512, 1024, stride=2, leaky=0.1),
            conv_dw(1024, 1024, stride=2, leaky=0.1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return ['lr', 'momentum']