import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def supported_hyperparameters():
    return {'lr', 'momentum', 'batch', 'transform'}


class InvertedBottleneck(nn.Module):
    """
    Inverted bottleneck block:
    1x1 expand -> BN -> SiLU -> 3x3 depthwise -> BN -> SiLU -> 1x1 project -> BN
    Optional stride on depthwise conv for downsampling.
    Residual connection when input/output shapes match.
    """
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.project(y)
        if self.use_residual:
            y = y + x
        return y


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w.unsqueeze(-1).unsqueeze(-1)


class MultiScaleHead(nn.Module):
    """
    Fuses features from multiple spatial scales via 1x1 convs + global pooling.
    """
    def __init__(self, ch_list, out_ch):
        super().__init__()
        self.branches = nn.ModuleList()
        for c in ch_list:
            self.branches.append(nn.Sequential(
                nn.Conv2d(c, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            ))
        self.classifier = nn.Linear(out_ch * len(ch_list), out_ch)

    def forward(self, feats):
        outs = []
        for i, f in enumerate(feats):
            outs.append(self.branches[i](f).flatten(1))
        x = torch.cat(outs, dim=1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.transform = prm.get('transform', 'norm_32_flip')

        in_ch = in_shape[1]
        n_classes = out_shape[0] if out_shape else 10

        # Stem: 3 -> 48
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )

        # Stage 1: 48 -> 48, two inverted bottleneck blocks, stride 1
        self.stage1 = nn.Sequential(
            InvertedBottleneck(48, 48, expand_ratio=4, stride=1),
            InvertedBottleneck(48, 48, expand_ratio=4, stride=1),
        )

        # Stage 2: 48 -> 96, two blocks, first with stride 2
        self.stage2 = nn.Sequential(
            InvertedBottleneck(48, 96, expand_ratio=4, stride=2),
            InvertedBottleneck(96, 96, expand_ratio=4, stride=1),
        )

        # Stage 3: 96 -> 192, two blocks, first with stride 2
        self.stage3 = nn.Sequential(
            InvertedBottleneck(96, 192, expand_ratio=4, stride=2),
            InvertedBottleneck(192, 192, expand_ratio=4, stride=1),
        )

        # Stage 4: 192 -> 384, two blocks, first with stride 2
        self.stage4 = nn.Sequential(
            InvertedBottleneck(192, 384, expand_ratio=4, stride=2),
            InvertedBottleneck(384, 384, expand_ratio=4, stride=1),
        )

        # SE attention on the final stage output
        self.se = SEBlock(384, reduction=4)

        # Multi-scale head: collect features from stages 2, 3, 4
        # stage2 output: 96 ch, stage3 output: 192 ch, stage4 output: 384 ch
        self.head = MultiScaleHead([96, 192, 384], out_ch=192)

        self.classifier = nn.Linear(192, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform in ('norm_32', 'norm_32_flip'):
            x = (x - 0.4914) / 0.2470

        x = self.stem(x)
        x = self.stage1(x)

        feat2 = self.stage2(x)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat4 = self.se(feat4)

        x = self.head([feat2, feat3, feat4])
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm['lr'],
            betas=(prm['momentum'], 0.999),
            weight_decay=1e-4,
        )
        self.batch_size = prm.get('batch', 128)
        return self.optimizer

    def learn(self, train_data):
        self.train()
        try:
            total_steps = len(train_data)
        except (TypeError, AttributeError):
            total_steps = 1000

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0.0,
        )

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            scheduler.step()