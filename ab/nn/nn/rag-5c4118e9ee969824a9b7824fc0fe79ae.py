# Auto-generated single-file for ConvRelPosEnc
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
def _assert(condition, message): assert condition, message

# ---- original imports from contributing modules ----

# ---- ConvRelPosEnc (target) ----
class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, head_chs, num_heads, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                1. An integer of window size, which assigns all attention heads with the same window s
                    size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits (
                    e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: num_heads}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            # Determine padding size.
            # Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * head_chs,
                cur_head_split * head_chs,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * head_chs,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * head_chs for x in self.head_splits]

    def forward(self, q, v, size: Tuple[int, int]):
        B, num_heads, N, C = q.shape
        H, W = size
        _assert(N == 1 + H * W, '')

        # Convolutional relative position encoding.
        q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]

        v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)

        EV_hat = q_img * conv_v_img
        EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
        return EV_hat

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
        self.conv_rel_pos = ConvRelPosEnc(head_chs=32, num_heads=1, window=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)

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
        B, C, H, W = x.shape
        
        q = torch.randn(B, 1, 1 + H * W, 32, device=x.device)
        v = torch.randn(B, 1, 1 + H * W, 32, device=x.device)
        
        pos_enc = self.conv_rel_pos(q, v, (H, W))
        
        x_flat = x.flatten(2).transpose(1, 2) 
        x_flat = x_flat + pos_enc.squeeze(1)[:, 1:, :] 
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

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
