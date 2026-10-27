import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # channels-first (C, H, W)
        num_classes = out_shape[0]  # number of target classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.build_layer(32, 32, 2, 1)
        self.layer2 = self.build_layer(32, 64, 2, 2)
        self.layer3 = self.build_layer(64, 128, 2, 2)
        self.layer4 = self.build_layer(128, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        self._is_fwd_comp = False

    def build_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )]
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self._is_fwd_comp:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.flatten(1)
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

    def compile(self, prm):
        self._is_fwd_comp = True
        self.input_resolution = prm.get('input_resolution', 224)
        self.num_classes = prm.get('num_classes', 1000)
        self.device = prm.get('device', 'cpu')
        self.in_channels = prm.get('in_channels', 3)
        self.kernel_size = prm.get('kernel_size', 7)
        self.head_dim = prm.get('head_dim', 64)
        self.num_heads = prm.get('num_heads', 12)
        self.hidden_dim = prm.get('hidden_dim', 3072)
        self.dropout = prm.get('dropout', 0.1)
        self.conv1_size = self.head_dim * self.num_heads
        self.conv1_stride = prm.get('conv1_stride', 4)
        self.conv1_padding = prm.get('conv1_padding', 0)
        self.conv1_groups = self.num_heads
        self.kernel_size = prm.get('kernel_size', 11)
        self.maxpool_kernel_size = prm.get('maxpool_kernel_size', 3)
        self.maxpool_stride = prm.get('maxpool_stride', 2)
        self.maxpool_padding = prm.get('maxpool_padding', 0)
        self.layer_norm_eps = prm.get('layer_norm_eps', 1e-06)
        self.layer_norm_elementwise_affine = prm.get('layer_norm_elementwise_affine', True)
        self.dropout = prm.get('dropout', 0.1)
        self.dropout_token = prm.get('dropout_token', 0.1)
        self.dropout_attn = prm.get('dropout_attn', 0.1)
        self.dropout_fused = prm.get('dropout_fused', 0.1)
        self.dropout_norm = prm.get('dropout_norm', 0.1)
        self.dropout_norm_attn = prm.get('dropout_norm_attn', 0.1)
        self.dropout_attn_fused = prm.get('dropout_attn_fused', 0.1)
        self.dropout_norm_fused = prm.get('dropout_norm_fused', 0.1)
        self.dropout_norm_fused_attn = prm.get('dropout_norm_fused_attn', 0.1)
        self.dropout_norm_fused_attn_fused = prm.get('dropout_norm_fused_attn_fused', 0.1)
        self.dropout_norm_fused_attn_fused_norm = prm.get('dropout_norm_fused_attn_fused_norm', 0.1)
        self.dropout_norm_fused_attn_fused_norm_attn = prm.get('dropout_norm_fused_attn_fused_norm_attn', 0.1)
        self.dropout_norm_fused_attn_fused_norm_attn_fused = prm.get('dropout_norm_fused_attn_fused_norm_attn_fused', 0.1)
        self.dropout_norm_fused_attn_fused_norm_attn_fused_norm = prm.get('dropout_norm_fused_attn_fused_norm_attn_fused_norm', 0.1)
        self.dropout_norm_fused_attn_fused_norm_attn_fused_norm_attn = prm.get('dropout_norm_fused_attn_fused_norm_attn_fused_norm_attn', 0.1)
        self.dropout_norm_fused_attn_fused_norm_attn_fused_norm_attn_fused = prm.get('dropout_norm_fused_attn_fused_norm_attn_fused_norm_attn_fused', 0.1)
        self.dropout_norm_fused_attn_fused_norm_attn_fused_norm_attn_fused_norm = prm.get('dropout_norm_fused_attn_fused_norm_attn_fused_norm_attn_fused_norm', 0.1)
        self.dropout_norm_fused_attn_