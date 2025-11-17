import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor


def supported_hyperparameters():
    return {"lr", "momentum"}


class EfficientChannelReduction(nn.Module):
    def __init__(self, input_channels: int, num_reduced_channels: Optional[int]):
        super().__init__()
        if num_reduced_channels is None or num_reduced_channels <= 0:
            num_reduced_channels = max(1, input_channels // 4)
        self.act = nn.ReLU(inplace=True)
        self.reduce = nn.Linear(input_channels, num_reduced_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(x)
        return self.reduce(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = max(1, int(groups))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        if self.groups <= 1 or c % self.groups != 0:
            return x
        g = self.groups
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


class DecoderAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        out, w = self.multihead_attn(queries, memory, memory, key_padding_mask=key_padding_mask)
        return out, w


class Net(nn.Module):
    def __init__(self, in_shape: Any, out_shape: Any, prm: dict, device: torch.device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.hidden_size = int(self.prm.get("hidden_size", 768))
        self.num_heads = int(self.prm.get("num_heads", 8))
        if self.hidden_size % self.num_heads != 0:
            self.hidden_size = max(self.num_heads, ((self.hidden_size // self.num_heads) + 1) * self.num_heads)
        self.vocab_size = self._first_int(out_shape)
        self.pad_idx = int(self.prm.get("pad_idx", 0))
        self.sos_idx = int(self.prm.get("sos_idx", 1))
        self.eos_idx = int(self.prm.get("eos_idx", 2))
        self.max_len = int(self.prm.get("max_len", 16))

        self.in_channels, self.in_h, self.in_w = self._infer_shape(in_shape)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, self.hidden_size, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(inplace=True),
        )

        self.memory_proj_key = nn.Linear(self.hidden_size, self.hidden_size)
        self.memory_proj_value = nn.Linear(self.hidden_size, self.hidden_size)

        embed_dim = int(self.prm.get("embed_dim", self.hidden_size))
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(embed_dim, self.hidden_size, batch_first=True, num_layers=1)
        self.attention = DecoderAttention(self.hidden_size, self.num_heads)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

        self.dropout = nn.Dropout(float(self.prm.get("dropout", 0.1)))

        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self._init_weights()
        self.to(self.device)

    @staticmethod
    def supported_hyperparameters():
        return {"lr", "momentum"}

    @staticmethod
    def _first_int(x: Any) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        return int(x)

    @staticmethod
    def _infer_shape(in_shape: Any) -> Tuple[int, int, int]:
        if isinstance(in_shape, (tuple, list)):
            if len(in_shape) >= 4:
                return int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
            if len(in_shape) == 3:
                return int(in_shape[0]), int(in_shape[1]), int(in_shape[2])
            if len(in_shape) == 2:
                return int(in_shape[0]), int(in_shape[1]), int(in_shape[1])
        return 3, 224, 224

    def _normalize_captions(self, captions: Tensor) -> Tensor:
        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 3:
            captions = captions[:, 0, :]
        return captions

    def _encode_image(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.encoder(images)                                      # [B, Hdim, H', W']
        B, C, Hp, Wp = feat.shape
        memory = feat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)         # [B, S, H]
        mem_k = self.memory_proj_key(memory)
        mem_v = self.memory_proj_value(memory)
        return mem_k, mem_v

    def train_setup(self, prm: dict):
        prm = prm or {}
        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        lr = float(prm.get("lr", 1e-3))
        beta1 = float(prm.get("momentum", 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        prm = getattr(train_data, "prm", self.prm)
        self.train_setup(prm)

        for batch in train_data:
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue
                images, captions = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get("x", None)
                captions = batch.get("y", None)
                if images is None or captions is None:
                    continue
            else:
                images = getattr(batch, "x", None)
                captions = getattr(batch, "y", None)
                if images is None or captions is None:
                    continue

            images = images.to(self.device)
            captions = self._normalize_captions(captions.to(self.device).long())
            if captions.size(1) <= 1:
                continue

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            logits, _ = self.forward(images, inputs)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def init_zero_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, images: Tensor, captions: Optional[Tensor] = None):
        images = images.to(self.device)
        mem_k, mem_v = self._encode_image(images)

        B = images.size(0)
        if captions is None:
            captions = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        else:
            captions = self._normalize_captions(captions.to(self.device).long())

        emb = self.dropout(self.embedding(captions))
        dec_out, _ = self.gru(emb)

        attn_out, attn_w = self.attention(dec_out, mem_v, key_padding_mask=None)
        logits = self.classifier(self.dropout(attn_out))
        return logits, attn_w

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        self.eval()
        images = images.to(self.device)
        mem_k, mem_v = self._encode_image(images)
        B = images.size(0)

        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        hidden = None

        for _ in range(self.max_len - 1):
            emb = self.embedding(tokens[:, -1:])
            dec_out, hidden = self.gru(emb, hidden)
            attn_out, _ = self.attention(dec_out, mem_v, key_padding_mask=None)
            logits = self.classifier(attn_out)          # [B,1,V]
            next_tok = logits.squeeze(1).argmax(dim=-1, keepdim=True)  # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == self.eos_idx).all():
                break

        return tokens

    def _init_weights(self):
        init_range = 0.02
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_range)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.memory_proj_key.weight)
        if self.memory_proj_key.bias is not None:
            nn.init.zeros_(self.memory_proj_key.bias)
        nn.init.xavier_uniform_(self.memory_proj_value.weight)
        if self.memory_proj_value.bias is not None:
            nn.init.zeros_(self.memory_proj_value.bias)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def model_net(in_shape: Any, out_shape: Any, prm: dict, device: torch.device):
    return Net(in_shape, out_shape, prm, device)
