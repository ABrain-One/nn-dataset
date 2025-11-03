import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Standard squeeze-excitation for 4D feature maps."""
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        assert ratio >= 1
        hidden = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)   # [B, C]
        w = self.fc(y).view(b, c, 1, 1)   # [B, C, 1, 1]
        return x * w


class ViTStyleEncoder(nn.Module):
    """
    Patchify + project + add learned positional embeddings.
    Returns a sequence of tokens: [B, S, D].
    """
    def __init__(self, in_channels: int, patch_size: int, hidden_size: int, max_hw: int = 224):
        super().__init__()
        assert patch_size > 0
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # A conservative cap for maximum number of patches (for pos embedding size)
        max_patches = (max_hw // patch_size) * (max_hw // patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, max_patches, hidden_size) * 0.02)

        # we project flattened patches (C * P * P) -> D
        self.proj = nn.Linear(in_channels * (patch_size ** 2), hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]; H,W are expected multiples of patch_size
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"H and W must be divisible by patch_size={P}"
        # Unfold to patches: [B, C*P*P, S] with S = (H/P)*(W/P)
        patches = F.unfold(x, kernel_size=P, stride=P)             # [B, C*P*P, S]
        patches = patches.transpose(1, 2)                           # [B, S, C*P*P]
        tokens = self.proj(patches)                                 # [B, S, D]

        # Positional embeddings (trim/pad as needed)
        S = tokens.size(1)
        pos = self.pos_emb[:, :S, :]                                # [1, S, D]
        return tokens + pos


class TransformerDecoder(nn.Module):
    """
    Wrapper around nn.TransformerDecoder that embeds target token ids and
    returns decoder hidden states. Generation is greedy.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int, vocab_size: int,
                 pad_idx: int = 0, max_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        # sinusoidal positions
        self.register_buffer("pos_cache", None, persistent=False)

    @staticmethod
    def _build_sin_pos(T: int, d_model: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(T, d_model, device=device)
        position = torch.arange(0, T, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [T, D]

    def _pos(self, B: int, T: int, d: int, device: torch.device) -> torch.Tensor:
        if self.pos_cache is None or self.pos_cache.size(0) < T or self.pos_cache.size(1) != d:
            self.pos_cache = self._build_sin_pos(T, d, device)
        return self.pos_cache[:T, :].unsqueeze(0).expand(B, -1, -1)  # [B, T, D]

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # For batch_first=True, attn_mask is [T, T]
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        # convert to float mask with -inf for masked positions
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, memory: torch.Tensor, tgt_ids: Optional[torch.Tensor],
                sos_idx: int, eos_idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        memory: [B, S, D]; tgt_ids: [B, T] or None
        Returns: (states/logits-ready) [B, T, D], and optionally generated ids if tgt_ids is None
        """
        B, S, D = memory.shape
        device = memory.device

        if tgt_ids is not None:
            T = tgt_ids.size(1)
            x = self.token_embed(tgt_ids) + self._pos(B, T, D, device)  # [B, T, D]
            causal = self._causal_mask(T, device)                        # [T, T]
            y = self.decoder(tgt=x, memory=memory, tgt_mask=causal)      # [B, T, D]
            return self.ln(y), None

        # Greedy generation
        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        states = []
        for _ in range(self.max_len):
            T = tokens.size(1)
            x = self.token_embed(tokens) + self._pos(B, T, D, device)
            causal = self._causal_mask(T, device)
            y = self.decoder(tgt=x, memory=memory, tgt_mask=causal)      # [B, T, D]
            y = self.ln(y)
            step_logits = y[:, -1, :]                                    # [B, D] (will be projected outside)
            states.append(step_logits.unsqueeze(1))                      # store hidden for projection
            # next token (softmax happens after projection in Net)
            # do a small linear probe with a temp projection assumption outside Net,
            # here we just pick a placeholder id (will be corrected in Net generation)
            # We return states for Net to project -> logits -> argmax.
            # So we stop here; Net will control the loop if needed.
            break

        # If used directly (one step only), return current states and tokens
        ycat = torch.cat(states, dim=1) if states else torch.zeros(B, 0, D, device=device)
        return ycat, tokens  # tokens only contains SOS here; Net controls full decode


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device

        # Resolve shapes
        self.in_channels = int(in_shape[1]) if isinstance(in_shape, (tuple, list)) else 3
        if isinstance(out_shape, (tuple, list)):
            self.vocab_size = int(out_shape[0])
        else:
            self.vocab_size = int(out_shape)

        # Hyperparams
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.patch_size  = int(prm.get("patch_size", 16))
        self.num_layers  = int(prm.get("num_layers", 4))
        self.num_heads   = int(prm.get("num_heads", 8))
        self.dropout     = float(prm.get("dropout", 0.1))
        self.max_len     = int(prm.get("max_len", 30))
        self.pad_idx     = int(prm.get("pad_idx", 0))
        self.sos_idx     = int(prm.get("sos_idx", 1))
        self.eos_idx     = int(prm.get("eos_idx", 2))

        if self.hidden_size % self.num_heads != 0:
            # pick a compatible head count
            for h in (12, 8, 6, 4, 3, 2):
                if self.hidden_size % h == 0:
                    self.num_heads = h
                    break

        # Encoder (ViT-style)
        self.encoder = ViTStyleEncoder(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            max_hw=int(prm.get("max_hw", 224)),
        )

        # Decoder (states only; Net will project to vocab)
        self.decoder = TransformerDecoder(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            vocab_size=self.vocab_size,
            pad_idx=self.pad_idx,
            max_len=self.max_len,
            dropout=self.dropout,
        )

        # Final projection to vocab
        self.proj = nn.Linear(self.hidden_size, self.vocab_size)

        # Train helpers
        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    # --- training API
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=prm["lr"],
            betas=(prm.get("momentum", 0.9), 0.999),
        )

    def learn(self, train_data):
        """train_data: iterable of (images, captions) with captions [B,T] or [B,1,T]"""
        if not self.criteria or self.optimizer is None:
            return
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            if captions.dim() == 3:
                captions = captions[:, 0, :]

            # teacher forcing (input =[:-1], target =[1:])
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            memory = self.encoder(images)  # [B, S, D]
            states, _ = self.decoder(memory, inputs, self.sos_idx, self.eos_idx)  # [B, T-1, D]
            logits = self.proj(states)                                            # [B, T-1, V]

            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    # --- inference/forward
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        If y is provided (teacher forcing):
            returns (logits [B, T-1, V], None)
        If y is None (greedy):
            returns (logits [B, Tgen, V], tokens [B, Tgen])
        """
        x = x.to(self.device)
        memory = self.encoder(x)  # [B, S, D]

        if y is not None:
            if y.dim() == 3:
                y = y[:, 0, :]
            y = y.to(self.device)
            if y.size(1) <= 1:
                empty = torch.zeros(y.size(0), 0, self.vocab_size, device=self.device)
                return empty, None
            inputs = y[:, :-1]
            states, _ = self.decoder(memory, inputs, self.sos_idx, self.eos_idx)  # [B, T-1, D]
            logits = self.proj(states)                                            # [B, T-1, V]
            return logits, None

        # Greedy decode fully here
        B = x.size(0)
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=self.device)
        logits_steps = []
        for _ in range(self.max_len):
            states, _ = self.decoder(memory, tokens, self.sos_idx, self.eos_idx)  # [B, T, D]
            step_logits = self.proj(states[:, -1:, :])                            # [B,1,V]
            logits_steps.append(step_logits)
            next_tok = step_logits.argmax(-1)                                     # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.eos_idx).all():
                break
        logits = torch.cat(logits_steps, dim=1) if logits_steps else torch.zeros(B, 0, self.vocab_size, device=self.device)
        # drop initial SOS in tokens for return (optional)
        return logits, tokens[:, 1:]


def supported_hyperparameters():
    return {"lr", "momentum"}


# -------- quick self-test (optional) --------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W, V = 2, 3, 224, 224, 5000
    imgs = torch.randn(B, C, H, W, device=device)
    caps = torch.randint(0, V, (B, 12), device=device)
    caps[:, 0] = 1  # SOS

    prm = dict(lr=1e-4, momentum=0.9, hidden_size=768, patch_size=16,
               num_layers=2, num_heads=8, dropout=0.1, max_len=16,
               pad_idx=0, sos_idx=1, eos_idx=2)

    net = Net((B, C, H, W), (V,), prm, device)
    net.train_setup(prm)

    # teacher forcing
    tf_logits, _ = net(imgs, caps)
    print("TF logits:", tf_logits.shape)  # [B, T-1, V]

    # greedy
    gen_logits, toks = net(imgs, None)
    print("Greedy:", gen_logits.shape, toks.shape)
