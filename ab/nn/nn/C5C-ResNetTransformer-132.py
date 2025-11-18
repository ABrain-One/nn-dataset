import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0] if out_shape.ndim == 2 else out_shape[0][-1]
        self.device = device

        # Encoder
        self.encoder = self.build_encoder(prm, in_shape, out_shape)

        # Decoder
        self.embed = nn.Embedding(self.vocab_size, 768)
        self.rnn = nn.LSTM(768, 768, batch_first=True)
        self.proj = nn.Linear(768, self.vocab_size)

    def build_encoder(self, prm, in_shape, out_shape):
        # Define a basic block
        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, stride=stride, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        # Build the encoder
        self.stem = nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        self.layer4 = self._make_layer(512, 1024, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1024, 768)

        return self

    def _make_layer(self, in_ch, out_ch, blocks):
        layer = []
        for _ in range(blocks):
            layer.append(BasicBlock(in_ch, out_ch))
        return nn.Sequential(*layer)

    def train_setup(self, optimizer: Optional[optim.Optimizer], lr: float, momentum: float):
        # Set up the model for training
        # We'll set the learning rate and momentum for the optimizer.
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if momentum is not None:
            for param_group in optimizer.param_groups:
                param_group['momentum'] = momentum

    def learn(self, images, captions):
        # Compute the loss
        images = images.to(self.device, dtype=torch.float32)
        if captions.ndim == 3:
            captions = captions[:,0,:].long().to(self.device)
        else:
            captions = captions.long().to(self.device)

        B = images.shape[0]
        T_prev = captions.shape[1] - 1

        memory = self.encoder(images)  # [B, 1, 768]
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        embedded = self.embed(inputs)  # [B, T_prev, 768]
        expanded_mem = memory.expand(B, T_prev, 768)
        embedded_cat = torch.cat([embedded, expanded_mem], -1)

        output, hidden = self.rnn(embedded_cat, None)
        logits = self.proj(output)

        # Flatten for cross entropy
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat)

        return loss

    def forward(self, images, captions):
        # Move images and captions to device
        images = images.to(self.device, dtype=torch.float32)
        if captions.ndim == 3:
            captions = captions[:,0,:].long().to(self.device)
        else:
            captions = captions.long().to(self.device)

        # Encode the images
        memory = self.encoder(images)  # [B, 1, 768]

        # Process the captions
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        embedded = self.embed(inputs)  # [B, T_prev, 768]
        B = images.shape[0]
        T_prev = inputs.shape[1]
        expanded_mem = memory.expand(B, T_prev, 768)
        embedded_cat = torch.cat([embedded, expanded_mem], -1)

        output, hidden = self.rnn(embedded_cat, None)
        logits = self.proj(output)

        # Assert shapes
        assert logits.shape[0] == images.shape[0]
        assert logits.shape[1] == inputs.shape[1]
        assert logits.shape[2] == self.vocab_size

        return logits, hidden[0]

def supported_hyperparameters():
    return {'lr', 'momentum'}