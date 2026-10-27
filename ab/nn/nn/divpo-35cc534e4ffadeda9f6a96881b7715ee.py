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
        self.layer1 = self.build_layer(32, 32, 2, stride=1)
        self.layer2 = self.build_layer(32, 64, 2, stride=2)
        self.layer3 = self.build_layer(64, 128, 2, stride=2)
        self.layer4 = self.build_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prm.get('dropout', 0.5)),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_shape[0]),
        )
        self._is_fwd_comp = False
        self._last_mu = None

    def build_layer(self, in_channels, out_channels, layers, stride):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(layers - 1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self._last_mu(x) if self._is_fwd_comp and self._last_mu is not None else x
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
            self.optimizer.step()

    def fuse_conv_bn(self, conv, bn):
        # https://github.com/ultralytics/yolov5/issues/29#issuecomment-643433142
        # https://github.com/ultralytics/yolov5/commit/bf8960f0c1c67f30f278f3181e0c367b3297b235
        bn = bn.eval()
        bn_w = bn.weight / torch.sqrt(bn.eps + bn.running_var)
        bn_b = bn.bias - bn.running_mean * bn_w
        conv.weight.data = conv.weight.data * bn_w.reshape([conv.out_channels, 1, 1, 1])
        conv.bias.data = conv.bias.data + bn_b
        return conv

    def forward_fuse(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def check_kernel_sizes(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                assert all(x in [3, 5, 7] for x in layer.kernel_size), layer

    def check_fpn_strides(self):
        for layer in self.modules():
            if isinstance(layer, nn.MaxPool2d):
                assert all(x in [8, 16, 32] for x in layer.kernel_size), layer

    def check_num_blocks(self):
        num_blocks = [3, 4, 6, 3]
        for layer in self.modules():
            if isinstance(layer, nn.Sequential):
                assert len(layer) in num_blocks, layer

    def check_heads(self):
        heads = [64, 128, 256, 512]
        for layer in self.modules():
            if isinstance(layer, nn.Sequential):
                assert all(x in [64, 128, 256, 512] for x in layer[0].in_channels), layer
            if isinstance(layer, nn.Linear):
                assert all(x in [64, 128, 256, 512] for x in layer.in_features), layer

    def check_fuse(self):
        assert self._is_fwd_comp
        assert isinstance(self._last_mu, nn.Sequential)

    def build_mobilenet_v3_small(self, in_shape):
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenet.py
        def conv_1x1_bn(inp, oup):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU()
            )

        def conv_3x3_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU()
            )

        def conv_1x1_bn_fuse(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

        def conv_3x3_bn_fuse(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup)
            )

        def se_block(inp, oup):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
                nn.SiLU(),
                nn.Conv2d(oup, oup, 1, 1, 0, bias=True),
                nn.Sigmoid()
            )

        def expand_conv_1x1_bn_fuse(inp, oup, stride, expand_ratio):
            out