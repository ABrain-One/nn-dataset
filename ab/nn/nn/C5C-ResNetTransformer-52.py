import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, 
                 activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = norm_layer(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                norm_layer(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        if self.shortcut:
            identity = self.shortcut(x)

        out += identity
        out = self.activation(out)

        return out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, key_padding_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        if memory is not None:
            tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear1(self.activation(self.norm2(tgt)))
        tgt2 = self.linear2(self.activation(tgt2))
        tgt = tgt + self.dropout(tgt2)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, d_inner, n_head, num_layers, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_inner)
        self.decoder = nn.TransformerDecoder(self.transformer, num_layers=num_layers)

        # Project the output to vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        # tgt: [B, T]
        # memory: [B, S, d_model]

        # Embedding: [B, T] -> [B, T, d_model]
        tgt = self.embedding(tgt)
        tgt_mask = self.generate_square_attention_mask(tgt.size(1)).to(tgt.device)
        memory_mask = None
        memory_key_padding_mask = None

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask)

        # Project to vocab_size
        output = self.proj(output)
        return output

    def generate_square_attention_mask(self, size):
        # Generate a square attention mask for the decoder
        mask = torch.triu(
            torch.ones((size, size), dtype=torch.bool),
            diagonal=1
        )
        return mask

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]

        # Define the encoder
        stem_width = 32
        self.stem = Conv2dNormActivation(in_shape[1], stem_width, kernel_size=3, stride=1, padding=1)

        # Define block parameters for exponential growth to reach 640
        self.stage1 = nn.Sequential(
            ResBottleneckBlock(stem_width, stem_width * 2, stride=1),
            ResBottleneckBlock(stem_width * 2, stem_width * 2, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage2 = nn.Sequential(
            ResBottleneckBlock(stem_width * 2, stem_width * 4, stride=1),
            ResBottleneckBlock(stem_width * 4, stem_width * 4, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage3 = nn.Sequential(
            ResBottleneckBlock(stem_width * 4, stem_width * 8, stride=1),
            ResBottleneckBlock(stem_width * 8, stem_width * 8, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage4 = nn.Sequential(
            ResBottleneckBlock(stem_width * 8, stem_width * 16, stride=1),
            ResBottleneckBlock(stem_width * 16, stem_width * 16, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage5 = nn.Sequential(
            ResBottleneckBlock(stem_width * 16, stem_width * 32, stride=1),
            ResBottleneckBlock(stem_width * 32, stem_width * 32, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Final stage: we want to output 640 channels
        self.stage6 = nn.Sequential(
            ResBottleneckBlock(stem_width * 32, 640, stride=1),
            ResBottleneckBlock(640, 640, stride=1)
        )

        # Now, the final feature map is 7x7 (if we started with 224) and 640 channels.
        # We then flatten it to a sequence of length 7*7=49.
        self.flatten = nn.Flatten()
        self.project = nn.Identity()

        # Define the decoder
        self.decoder = TransformerDecoder(
            d_model=640,
            d_inner=640 * 4,
            n_head=8,
            num_layers=6,
            vocab_size=self.vocab_size
        )

    def train_setup(self, prm):
        # This function is used to set up the model for training, e.g., setting the device.
        pass

    def learn(self, x, y):
        # This function is used to update the model parameters, but in our case, we are using standard training.
        pass

    def forward(self, x, y):
        # x: images [B, C, H, W]
        # y: captions [B, T]

        # Pass through the encoder
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)

        # Now, x is [B, 640, 7, 7]
        memory = self.flatten(x)  # [B, 640*7*7] -> [B, 49*640]
        memory = memory.view(memory.size(0), -1, 640)  # [B, 49, 640]

        # Project if necessary (if the final stage output is not 640)
        # But we already have 640, so no need.

        output = self.decoder(y, memory)

        return output

def supported_hyperparameters():
    return {'lr','momentum'}