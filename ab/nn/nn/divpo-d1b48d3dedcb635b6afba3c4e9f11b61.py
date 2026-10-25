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
    def __init__(self, anchors, num_classes, in_size):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.size = in_size
        self.ignore_thresh = 0.5
        self.lambda_coord = 5.0

    def _loc_target(self, anchor, bbox):
        gx = ((bbox[0] + bbox[2]) * 0.5 - anchor[0]) / anchor[0]
        gy = ((bbox[1] + bbox[3]) * 0.5 - anchor[1]) / anchor[1]
        gw = (bbox[2] - bbox[0]) / anchor[0]
        gh = (bbox[3] - bbox[1]) / anchor[1]
        return torch.stack([gx, gy, gw, gh]).unsqueeze(dim=0)

    def _conf_target(self, anchor, bbox):
        bbox_xy = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        iou = bbox_iou(anchor, bbox, x1y1x2y2=False)
        conf = iou > self.ignore_thresh
        iou_loss = iou[conf]
        conf_loss = conf.long()
        return iou_loss, conf_loss

    def forward(self, pred_tensor, target_tensor):
        anchor = self.anchors
        pred_tensor = pred_tensor.view(-1, self.size, self.bbox_attrs)

        mask = target_tensor[..., 4] > 0
        mask_ignore = target_tensor[..., 4] < 0
        # loc_p = pred_tensor[mask].view(-1, 4)
        # loc_pi = pred_tensor[mask_ignore].view(-1, 4)
        # loc_t = self._loc_target(anchor, target_tensor[mask].view(-1, 4))
        # loc_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # conf_p = pred_tensor[mask][:, 4:]
        # conf_pi = pred_tensor[mask_ignore][:, 4:]
        # conf_t = self._conf_target(anchor, target_tensor[mask].view(-1, 4))
        # conf_loss = F.cross_entropy(conf_p, conf_t.long(), reduction='sum')

        # conf_loss += F.cross_entropy(-conf_pi, conf_t.long(), reduction='sum')
        # conf_loss /= pred_tensor.size(0)
        # loc_loss = loc_loss / pred_tensor.size(0)

        # return conf_loss + self.lambda_coord * loc_loss
        return pred_tensor

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        self.num_classes = out_shape[0]
        self.model = self.build_model(self.in_channels, self.num_classes)

    def build_model(self, in_channels: int, num_classes: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv_bn(32, 32, stride=2),
            conv_dw(32, 32, stride=1),
            conv_dw(32, 64, stride=2),
            conv_dw(64, 64, stride=1),
            conv_dw(64, 128, stride=2),
            conv_dw(128, 128, stride=1),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def supported_hyperparameters(self):
        return {'lr', 'momentum'}