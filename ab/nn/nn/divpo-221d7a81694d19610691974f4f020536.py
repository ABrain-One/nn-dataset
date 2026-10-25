import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 3, stride=1)
        self.layer2 = self.build_layer(32, 64, 3, stride=2)
        self.layer3 = self.build_layer(64, 128, 3, stride=2)
        self.layer4 = self.build_layer(128, 256, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0])
        )
        self._is_fwd_comitted = False

    
    def build_layer(self, in_channels, out_channels, kernel_size, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def _get_iou(self, bbox1, bbox2):
        bbox1_x1 = bbox1[0]
        bbox1_y1 = bbox1[1]
        bbox1_x2 = bbox1[2]
        bbox1_y2 = bbox1[3]

        bbox2_x1 = bbox2[0]
        bbox2_y1 = bbox2[1]
        bbox2_x2 = bbox2[2]
        bbox2_y2 = bbox2[3]

        iou_x1 = max(bbox1_x1, bbox2_x1)
        iou_y1 = max(bbox1_y1, bbox2_y1)
        iou_x2 = min(bbox1_x2, bbox2_x2)
        iou_y2 = min(bbox1_y2, bbox2_y2)

        iou_width_height = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
        bbox1_width_height = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
        bbox2_width_height = (bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1)

        iou = iou_width_height / (bbox1_width_height + bbox2_width_height - iou_width_height)
        return iou

    def forward_wrapper(self, x):
        if self._is_fwd_comitted:
            return self.forward(x)
        else:
            self._is_fwd_comitted = True
            return self.forward(x)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

def supported_hyperparameters():
    return ['lr', 'momentum']

def check_shape(x, y, channel_last):
    if channel_last:
        assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2] and x.shape[3] == y.shape[3], 'channel last'
    else:
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] and x.shape[2] == y.shape[2], 'channel first'

def ConvGNReLU(in_channels, out_channels, kernel_size, stride, padding, channel_last):
    if channel_last:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k,