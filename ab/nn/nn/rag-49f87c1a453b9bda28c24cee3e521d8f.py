# Auto-generated single-file for DyConv
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable

# ---- mmdet.models.dense_heads.atss_vlfusion_head.DyReLU ----
class DyReLU(nn.Module):
    """Dynamic ReLU."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expand_ratio = expand_ratio
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // expand_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // expand_ratio,
                      out_channels * self.expand_ratio),
            nn.Hardsigmoid(inplace=True))

    def forward(self, x) -> Tensor:
        x_out = x
        b, c, h, w = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x).view(b, -1, 1, 1)

        a1, b1, a2, b2 = torch.split(x, self.out_channels, dim=1)
        a1 = (a1 - 0.5) * 2 + 1.0
        a2 = (a2 - 0.5) * 2
        b1 = b1 - 0.5
        b2 = b2 - 0.5
        out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        return out

# ---- DyConv (target) ----
class DyConv(nn.Module):
    """Dynamic Convolution."""

    def __init__(self,
                 conv_func: Callable,
                 in_channels: int,
                 out_channels: int,
                 use_dyfuse: bool = True,
                 use_dyrelu: bool = False,
                 use_dcn: bool = False):
        super().__init__()

        self.dyconvs = nn.ModuleList()
        self.dyconvs.append(conv_func(in_channels, out_channels, 1))
        self.dyconvs.append(conv_func(in_channels, out_channels, 1))
        self.dyconvs.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.attnconv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = nn.Hardsigmoid(inplace=True)
        else:
            self.attnconv = None

        if use_dyrelu:
            self.relu = DyReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_dcn:
            self.offset = nn.Conv2d(
                in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.dyconvs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.attnconv is not None:
            for m in self.attnconv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs: dict) -> dict:
        visual_feats = inputs['visual']

        out_vis_feats = []
        for level, feature in enumerate(visual_feats):

            offset_conv_args = {}
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                offset_conv_args = dict(offset=offset, mask=mask)

            temp_feats = [self.dyconvs[1](feature, **offset_conv_args)]

            if level > 0:
                temp_feats.append(self.dyconvs[2](visual_feats[level - 1],
                                                  **offset_conv_args))
            if level < len(visual_feats) - 1:
                temp_feats.append(
                    F.upsample_bilinear(
                        self.dyconvs[0](visual_feats[level + 1],
                                        **offset_conv_args),
                        size=[feature.size(2),
                              feature.size(3)]))
            mean_feats = torch.mean(
                torch.stack(temp_feats), dim=0, keepdim=False)

            if self.attnconv is not None:
                attn_feat = []
                res_feat = []
                for feat in temp_feats:
                    res_feat.append(feat)
                    attn_feat.append(self.attnconv(feat))

                res_feat = torch.stack(res_feat)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_feat))

                mean_feats = torch.mean(
                    res_feat * spa_pyr_attn, dim=0, keepdim=False)

            out_vis_feats.append(mean_feats)

        out_vis_feats = [self.relu(item) for item in out_vis_feats]

        features_dict = {'visual': out_vis_feats, 'lang': inputs['lang']}

        return features_dict

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        self.dyconv = DyConv(
            conv_func=nn.Conv2d,
            in_channels=32,
            out_channels=32,
            use_dyfuse=True,
            use_dyrelu=False,
            use_dcn=False
        )

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
