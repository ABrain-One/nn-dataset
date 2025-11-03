import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------------------------
# Encoder building blocks
# -------------------------
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, se_reduction: float = 0.5):
        super().__init__()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Squeeze-and-Excitation
        if se_reduction:
            if se_reduction < 1.0:
                reduced = max(1, int(round(out_channels * se_reduction)))
            else:
                reduced = max(1, out_channels // int(round(se_reduction)))
            self.se_fc1 = nn.Linear(out_channels, reduced, bias=False)
            self.se_fc2 = nn.Linear(reduced, out_channels, bias=False)
        else:
            self.se_fc1 = None
            self.se_fc2 = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.block(x)

        if self.se_fc1 is not None:
            # Squeeze
            w = F.adaptive_avg_pool2d(out, 1).flatten(1)  # [B, C]
            # Excite
            w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(w), inplace=True)))  # [B, C]
            out = out * w.view(out.size(0), out.size(1), 1, 1)

        out = self.relu(out + identity)
        return out


# -------------------------
# Positional encoding (batch-first)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)  # (1, T, E)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, E)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# -------------------------
# Transformer-based decoder
# -------------------------
class TransformerBasedDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout, max_len=500)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,  # work with (B, T, E)
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # Project image features to decoder hidden size if needed
        self.memory_proj = nn.Linear(feature_dim, hidden_size) if feature_dim != hidden_size else nn.Identity()

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # (T, T) with -inf above diagonal for causal masking
        mask = torch.full((sz, sz), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_seq: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        input_seq: (B, T) token ids
        memory:    (B, S, F) encoder features (S can be 1)
        """
        tgt = self.embedding(input_seq)             # (B, T, H)
        tgt = self.pos_encoding(tgt)                # (B, T, H)

        if memory.dim() == 2:
            memory = memory.unsqueeze(1)            # (B, 1, F)
        memory = self.memory_proj(memory)           # (B, S, H)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)  # (T, T)
        dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)         # (B, T, H)
        logits = self.fc_out(dec_out)               # (B, T, V)
        return logits


# -------------------------
# Full model
# -------------------------
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.feature_dim = 768
        self.hidden_size = 768
        self.num_layers = 1
        self.enc_num_heads = 8

        # Robustly extract vocab size from possibly nested tuples/lists
        def _first_int(x):
            return _first_int(x[0]) if isinstance(x, (tuple, list)) else int(x)

        self.vocab_size = _first_int(out_shape)

        # Encoder: lightweight CNN + SE-style residuals
        channels_list = [64, 128, 256, 512]
        in_channels = in_shape[1] if len(in_shape) > 1 else 3
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_list[0]),
            nn.ReLU(inplace=True),
            EnhancedResidualBlock(channels_list[0], channels_list[1], stride=2, se_reduction=0.5),
            EnhancedResidualBlock(channels_list[1], channels_list[2], stride=2, se_reduction=0.5),
            EnhancedResidualBlock(channels_list[2], channels_list[3], stride=2, se_reduction=0.5),
        )
        self.adap_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(channels_list[-1], self.feature_dim)

        # Decoder
        self.decoder = TransformerBasedDecoder(
            vocab_size=self.vocab_size,
            feature_dim=self.feature_dim,
            hidden_size=self.hidden_size,
            num_heads=self.enc_num_heads,
            num_layers=self.num_layers,
            dropout=float(prm.get("dropout", 0.1)),
        )

        # Training helpers (can be overridden by train_setup)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(prm.get("lr", 1e-4)), betas=(float(prm.get("momentum", 0.9)), 0.999)
        )

        self.to(self.device)

    # Optional API for compatibility with some trainers
    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(2, batch, self.hidden_size, device=device)
        c0 = torch.zeros(2, batch, self.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Encode images -> (B, 1, feature_dim)
        feats = self.cnn(images)                               # (B, C, H, W)
        feats = self.adap_avg_pool(feats).flatten(1)           # (B, C)
        feats = self.projection(feats)                         # (B, F)
        memory = feats.unsqueeze(1)                            # (B, 1, F)

        if captions is not None:
            # Teacher forcing
            inputs = captions[:, :-1]                          # (B, T-1)
            logits = self.decoder(inputs, memory)              # (B, T-1, V)
            return logits, hidden_state
        else:
            # Greedy decode
            B = images.size(0)
            ys = torch.full((B, 1), 1, dtype=torch.long, device=self.device)  # SOS=1
            logits_steps = []
            for _ in range(20):  # max length
                step_logits = self.decoder(ys, memory)[:, -1:, :]  # (B, 1, V)
                logits_steps.append(step_logits)
                next_ids = step_logits.argmax(dim=-1)              # (B, 1)
                ys = torch.cat([ys, next_ids], dim=1)
                if (next_ids == (self.vocab_size - 1)).all():      # EOS assumed last id
                    break
            logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.empty(B, 0, self.vocab_size, device=self.device)
            return logits, hidden_state

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-4)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data) -> None:
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)

            logits, _ = self.forward(images, captions)  # (B, T-1, V)
            targets = captions[:, 1:]                   # (B, T-1)

            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()


# -------------------------
# Lightweight smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 128, 128
    V = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(in_shape=(B, C, H, W), out_shape=(V,), prm={"lr": 1e-4, "momentum": 0.9}, device=device)

    x = torch.randn(B, C, H, W, device=device)
    y = torch.randint(0, V, (B, 10), device=device)

    model.train_setup({"lr": 1e-4, "momentum": 0.9})
    logits, _ = model(x, y)
    print("Teacher-forced logits:", logits.shape)  # (B, T-1, V)

    logits_inf, _ = model(x, None)
    print("Greedy logits:", logits_inf.shape)      # (B, <=20, V)
