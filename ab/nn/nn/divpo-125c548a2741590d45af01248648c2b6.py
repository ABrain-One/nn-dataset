import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device

        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        num_classes = out_shape[0]
        self.stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.block1 = Block(32, 32)
        self.block2 = Block(32, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 64)
        self.block5 = Block(64, 64)
        self.block6 = Block(64, 64)
        self.block7 = Block(64, 64)
        self.block8 = Block(64, 64)
        self.block9 = Block(64, 64)
        self.block10 = Block(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

        self.to_device(device)
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

    def build_optimizer(self, prm, params):
        return torch.optim.SGD(
            params,
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def to_device(self, device):
        self.to(device)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to_device(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = self.build_optimizer(prm, self.parameters())

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

    def profile(self):
        x = torch.zeros(1, *self.stem.in_shape)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool(x)
        x = self.avgpool(x)
        macs = self.flops_counter(x)
        params = self.params_counter(self.stem)
        for b in [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7, self.block8, self.block9, self.block10]:
            params += self.params_counter(b)
            macs += self.flops_counter(b(x.clone()))
        return params, macs

    def flops_counter(self, x):
        """
        Estimate MACs (multiply-adds) for a single forward pass on an example of size x.
        """
        # https://github.com/pytorch/pytorch/issues/29748#issuecomment-640225623
        # https://github.com/facebookresearch/fb.resnet.torch/issues/13
        # https://github.com/pytorch/pytorch/issues/4852#issuecomment-425496047
        # https://github.com/pytorch/pytorch/issues/6055
        # https://github.com/pytorch/pytorch/issues/6055
        def _compute_per_layer_macs(m, x):
            if isinstance(m, Block):
                # m is a Block
                # block takes (b, c, h, w) => (b*h*w, c, 3, 3) => 9*c*3*3 = 81*c
                # (32, 32, 3, 3) => 32*32*3*3 = 864
                # (64, 32, 3, 3) => 64*32*3*3 = 1728
                # (64, 64, 3, 3) => 64*64*3*3 = 12288
                # (64, 64, 3, 3) => 64*64*3*3 = 12288
                return x[0].numel() * m.conv.in_channels * m.conv.kernel_size[0] * m.conv.kernel_size[1]
            elif isinstance(m, nn.Linear):
                # m is a Linear layer
                # Linear takes (b, in, out) => (b, in, 1) => in*1*out = out*in
                return x[0].numel() * m.in_features * m.out_features
            else:
                # m is an elementwise operation or other non-trainable layer
                # Assume an average cost of 2 flops per element
                return x[0].numel() * 2

        _hooks = []
        _hooks.append(x[0].register_hook(lambda grad: setattr(self, 'last_layer_grad', grad)))
        _hooks.append(x[0].register_hook(lambda x_: setattr(self, 'last_layer_x', x_)))
        _hooks.append(x[0].register_hook(lambda _: setattr(self, 'last_layer_hook_called', True)))

        _hook = x[0].register_forward_hook(_compute_per_layer_macs)
        _hooks.append(_hook)
        _hooks.append(self.avgpool.register_forward_hook(lambda m, *_args, **_kwargs: setattr(self, 'last_pool_hook_called', True)))

        _ = self(x)

        # https://github.com/pytorch/pytorch/issues/29748#issuecomment-640225623
        # https://github.com/facebookresearch/fb.