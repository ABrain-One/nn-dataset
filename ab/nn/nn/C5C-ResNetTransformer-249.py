import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class Conv2dNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EncoderCNN(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 768):
        super().__init__()
        self.stem = Conv2dNormAct(in_channels, 64, 3, 1, 1)
        self.layer1 = Conv2dNormAct(64, 128, 3, 1, 1)
        self.layer2 = Conv2dNormAct(128, 256, 3, 1, 1)
        self.layer3 = Conv2dNormAct(256, 512, 3, 1, 1)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.globalpool(x)
        feat = self.proj(x.flatten(1))  # [B, H]
        return feat.unsqueeze(1)        # [B, 1, H]


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_ff: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, hidden_state=None):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)

        seq_len = embedded.size(1)
        tgt_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=embedded.device),
            diagonal=1,
        )
        logits = self.transformer(tgt=embedded, memory=memory, tgt_mask=tgt_mask)
        logits = self.fc_out(logits)
        hidden_state = memory
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0] if isinstance(out_shape, tuple) else int(out_shape)

        if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1:
            in_channels = int(in_shape[1])
        else:
            in_channels = 3

        self.encoder = EncoderCNN(in_channels=in_channels, hidden_dim=768)
        self.decoder = DecoderTransformer(vocab_size=self.vocab_size, hidden_dim=768)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None

    def init_zero_hidden(self, batch, device):
        return None

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        lr = float(prm["lr"])
        momentum = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(momentum, 0.999),
        )

    def _norm_caps(self, caps: torch.Tensor) -> torch.Tensor:
        if caps.ndim == 3:
            if caps.size(1) == 1:
                caps = caps[:, 0, :]
            else:
                caps = caps[:, :, 0]
        return caps.long()

    def learn(self, train_data):
        assert self.optimizer is not None
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            captions = captions.to(self.device, dtype=torch.long)
            captions = self._norm_caps(captions)

            memory = self.encoder(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.decoder(inputs, memory)
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)

        if captions is not None:
            captions = captions.to(self.device, dtype=torch.long)
            captions = self._norm_caps(captions)
            inputs = captions[:, :-1]
            if hidden_state is None:
                hidden_state = memory
            logits, hidden_state = self.decoder(inputs, memory, hidden_state)
            return logits, hidden_state

        raise NotImplementedError("Inference without captions is not implemented")
