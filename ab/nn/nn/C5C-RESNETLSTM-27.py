import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ResNetBase(nn.Module):
    def __init__(self, in_shape, hidden_size, device):
        super().__init__()
        in_ch = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(64, hidden_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)                 # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)           # [B, 64] (keep batch dim)
        x = self.proj(x)                    # [B, hidden_size]
        return x


class ResNetSpatialEncoder(nn.Module):
    def __init__(self, in_shape, hidden_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn = ResNetBase(in_shape, hidden_size, device)

    def forward(self, images):
        return self.cnn(images)              # [B, hidden_size]


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(int(vocab_size), hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, int(vocab_size))

    def init_zero_hidden(self, batch, device):
        # GRU expects (num_layers, batch, hidden_size)
        return torch.zeros(1, batch, self.hidden_size, device=device)

    def forward(self, inputs, hidden_state=None, features=None):
        # inputs: [B, T]
        B, _ = inputs.shape
        device = inputs.device
        if hidden_state is None:
            hidden_state = self.init_zero_hidden(B, device)
        embedded = self.embedder(inputs)     # [B, T, H]
        output, hidden_state = self.gru(embedded, hidden_state)  # output: [B, T, H]
        logits = self.projection(output)     # [B, T, V]
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm or {}

        # Robustly extract vocab size (handles int, (V,), or nested like ((V,), ...))
        def _vsize(shape):
            if isinstance(shape, int):
                return int(shape)
            if isinstance(shape, (tuple, list)) and len(shape) > 0:
                first = shape[0]
                if isinstance(first, (tuple, list)) and len(first) > 0:
                    return int(first[0])
                return int(first)
            return int(shape)

        self.vocab_size = _vsize(out_shape)
        hidden_size = 768

        self.encoder = ResNetSpatialEncoder(in_shape, hidden_size, device)
        self.decoder = LSTMDecoder(self.vocab_size, hidden_size, device)

        # training bits (set in train_setup)
        self.criteria = None
        self.optimizer = None

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data):
        # Expect an iterable of (images, captions) batches
        if self.criteria is None or self.optimizer is None:
            self.train_setup(self.prm)

        criterion = self.criteria[0]
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # teacher forcing: predict next token
            features = self.encoder(images)
            logits, _ = self.decoder(captions[:, :-1], features=features)
            loss = criterion(logits.reshape(-1, self.vocab_size), captions[:, 1:].reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
        features = self.encoder(images)  # [B, H]

        if captions is not None:
            captions = captions.to(self.device)
            logits, hidden_state = self.decoder(captions, hidden_state, features=features)
            return logits, hidden_state

        # Inference path (no decoding loop provided here)
        return features
