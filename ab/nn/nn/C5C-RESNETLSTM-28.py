import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        # ---- API aliases (auto-injected, consolidated) ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3

        # Robust vocab size extraction: supports int, (V,), ((V,), ...)
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
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # ---------------- Encoder ----------------
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 11, 4, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, 3, 2, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, 3, 2, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 384, 3, 2, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, 2, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, 768)  # project pooled CNN features to 768

        # ---------------- Decoder ----------------
        self.embedding = nn.Embedding(self.vocab_size, 768)
        # Input to LSTM is [embed=768 || visual=768] = 1536
        self.rnn = nn.LSTM(768 + 768, 768, batch_first=True)
        self.fc_out = nn.Linear(768, self.vocab_size)

        # training helpers
        self.criteria = None
        self.optimizer = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def init_zero_hidden(self, batch_size, device):
        # LSTM expects (num_layers, batch, hidden_size) for both h_0 and c_0
        h0 = torch.zeros(1, batch_size, 768, device=device)
        c0 = torch.zeros(1, batch_size, 768, device=device)
        return (h0, c0)

    def _encode(self, images):
        # images: [B, C, H, W] -> features: [B, 768]
        x = self.cnn_encoder(images)
        x = self.global_pool(x)          # [B, 256, 1, 1]
        x = x.flatten(1)                 # [B, 256]
        x = self.proj(x)                 # [B, 768]
        return x

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
        feats = self._encode(images)     # [B, 768]

        if captions is not None:
            captions = captions.to(self.device)          # [B, L]
            emb = self.embedding(captions)               # [B, L, 768]

            # tile visual features across sequence length and concat with embeddings
            B, L, _ = emb.shape
            feats_seq = feats.unsqueeze(1).expand(B, L, feats.size(-1))  # [B, L, 768]
            rnn_in = torch.cat([emb, feats_seq], dim=-1)                 # [B, L, 1536]

            if hidden_state is None:
                hidden_state = self.init_zero_hidden(B, self.device)

            out, hidden_state = self.rnn(rnn_in, hidden_state)  # out: [B, L, 768]
            logits = self.fc_out(out)                           # [B, L, V]
            return logits, hidden_state

        # inference path (feature extractor output)
        return feats, None

    def learn(self, train_data):
        # Expect an iterable of (images, captions)
        if self.criteria is None or self.optimizer is None:
            self.train_setup({})  # fallback if not called

        criterion = self.criteria[0]
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # teacher forcing: predict next token
            logits, _ = self.forward(images, captions[:, :-1])          # [B, L-1, V]
            loss = criterion(logits.reshape(-1, self.vocab_size),        # flatten to [B*(L-1), V]
                              captions[:, 1:].reshape(-1))               # targets [B*(L-1)]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
