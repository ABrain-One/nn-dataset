import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {"lr", "momentum"}


# ----------------------------
# Attention / Utility modules
# ----------------------------
class ChannelReductionAttention(nn.Module):
    """
    Channel attention using a simple squeeze/excitation with a reduction ratio.
    Kept under the original class name; redesigned to be correct and executable.
    """
    def __init__(self, channels: int, ratio: int = 4, batch_norm: bool = True, device: Optional[torch.device] = None):
        super().__init__()
        reduced = max(1, channels // ratio)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # args kept for API compatibility
        self._use_bn = bool(batch_norm)
        self._device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.avg(x))
        return x * w


class SpatialChannelBottleneck(nn.Module):
    """
    Bottleneck that mixes spatial 1x1 conv + channel attention.
    Kept under the original class name; safe and executable.
    """
    def __init__(self, in_channels: int, out_channels: int, se_ratio: int = 4):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = ChannelReductionAttention(out_channels, ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spatial(x)
        y = self.se(y)
        return y


# ----------------------------
# Positional encoding (Transformer)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                 # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)       # even
        pe[:, 1::2] = torch.cos(position * div_term)       # odd
        pe = pe.unsqueeze(0)                                # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


# ----------------------------
# Encoder and Decoder
# ----------------------------
class EncoderCNN(nn.Module):
    """
    CNN encoder that produces a single image token (vector) per image.
    Combines a light stem with a few SpatialChannelBottleneck blocks,
    then global-average-pool and project to hidden size.
    """
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block1 = SpatialChannelBottleneck(128, 256)
        self.block2 = SpatialChannelBottleneck(256, 256)
        self.block3 = SpatialChannelBottleneck(256, 256)
        self.chan_attn = ChannelReductionAttention(256, ratio=4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, hidden_size]
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.chan_attn(x)
        x = self.pool(x).flatten(1)        # [B, 256]
        x = self.proj(x)                   # [B, hidden_size]
        return x


class CaptionTransformerDecoder(nn.Module):
    """
    Transformer decoder (embedding + positional enc + TransformerDecoder + output projection).
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.posenc = PositionalEncoding(hidden_size, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=4 * hidden_size,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # True where future positions are masked
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: [B, T] (token ids)
        memory:     [B, M, H] (e.g., M=1 image token)
        returns logits: [B, T, V]
        """
        B, T = tgt_tokens.shape
        x = self.embed(tgt_tokens)              # [B, T, H]
        x = self.posenc(x)                      # [B, T, H]
        tgt_mask = self.subsequent_mask(T, x.device)  # [T, T], bool
        dec = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)  # [B, T, H]
        logits = self.out_proj(dec)             # [B, T, V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor, max_len: int, sos_idx: int, eos_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        memory:  [B, M, H]
        returns:
          tokens: [B, T_gen]
          logits: [B, T_gen, V]
        """
        B = memory.size(0)
        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=memory.device)
        all_logits = []
        for _ in range(max_len):
            logits = self.forward(tokens, memory)     # [B, t, V]
            step_logits = logits[:, -1:, :]           # [B, 1, V]
            all_logits.append(step_logits)
            next_token = step_logits.argmax(-1)       # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token.squeeze(1) == eos_idx).all():
                break
        return tokens, torch.cat(all_logits, dim=1) if all_logits else torch.zeros(B, 0, self.out_proj.out_features, device=memory.device)


# ----------------------------
# Main Net (image captioning)
# ----------------------------
class Net(nn.Module):
    """
    Executable image captioning model with:
      - EncoderCNN -> image embedding
      - Light Transformer encoder on the image token (kept minimal)
      - CaptionTransformerDecoder for language modeling
    Original helper class names are preserved; errors removed; runnable end-to-end.
    """
    def __init__(self, in_shape: Tuple[int, ...], out_shape: Tuple[Union[int, Tuple], ...], prm: dict, device: torch.device):
        super().__init__()
        self.device = device

        # ---- Shapes ----
        if len(in_shape) == 4:
            in_channels = int(in_shape[1])
        elif len(in_shape) == 3:
            in_channels = int(in_shape[0])
        else:
            raise ValueError(f"in_shape should be (C,H,W) or (B,C,H,W), got {in_shape}")

        self.vocab_size = self._first_int(out_shape)

        # ---- Hyperparameters ----
        self.hidden_size = int(prm.get("hidden_size", 768))
        self.num_heads = int(prm.get("num_heads", 8))
        self.num_dec_layers = int(prm.get("num_decoding_layers", 1))
        self.enc_layers = int(prm.get("enc_layers", 1))           # light transformer encoder on image token
        self.dropout = float(prm.get("decoder_dropout", 0.1))
        self.pad_idx = int(prm.get("pad_idx", 0))
        self.sos_idx = int(prm.get("sos_idx", 1))
        self.eos_idx = int(prm.get("eos_idx", 2))
        self.max_len = int(prm.get("max_len", 20))

        # ---- Encoder CNN -> image token ----
        self.img_encoder = EncoderCNN(in_channels, self.hidden_size)

        # ---- (Optional) transformer encoder over the single token (kept for structure completeness) ----
        # With one token, this is effectively a LayerNorm stack; still included to match prior structure.
        enc_block = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=max(1, min(self.num_heads, self.hidden_size // 64)),
            dim_feedforward=4 * self.hidden_size, dropout=self.dropout, batch_first=True
        )
        self.img_token_encoder = nn.TransformerEncoder(enc_block, num_layers=max(1, self.enc_layers))

        # ---- Transformer decoder ----
        self.decoder = CaptionTransformerDecoder(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_dec_layers,
            dropout=self.dropout,
            pad_idx=self.pad_idx,
        )

        # training helpers
        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.grad_clip = float(prm.get("grad_clip", 3.0))

        self.to(self.device)

    # ---- API / helpers ----
    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Cannot infer integer vocab size from out_shape={x}")

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stub for API compatibility
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm["lr"], betas=(prm.get("momentum", 0.9), 0.999)
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """
        train_data: iterable of (images, captions)
          images:   [B, C, H, W]
          captions: [B, T] integer token ids, with SOS at index 0, PAD=self.pad_idx
        """
        if not self.criteria or self.optimizer is None:
            raise RuntimeError("Call train_setup(prm) before learn().")

        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            inp = captions[:, :-1]  # teacher forcing inputs
            tgt = captions[:, 1:]   # targets

            logits, _ = self.forward(images, inp)  # [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

            yield float(loss.detach().cpu())

    # ---- Forward ----
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        If captions is provided (teacher forcing):
           images:   [B,C,H,W]
           captions: [B,T] token ids (input to decoder)
           returns:  (logits [B,T,V], hidden_stub)

        If captions is None (greedy):
           returns:  (logits [B,T_gen,V], hidden_stub)
        """
        B = images.size(0)

        # Encode image to a single token
        img_vec = self.img_encoder(images)     # [B, H]
        memory = img_vec[:, None, :]           # [B, 1, H]

        # Optional transformer encoder (over single token)
        memory = self.img_token_encoder(memory)

        hidden_stub = self.init_zero_hidden(B, images.device)

        if captions is not None:
            # Teacher forcing path
            logits = self.decoder(captions, memory)  # [B, T, V]
            return logits, hidden_stub

        # Greedy generation
        tokens, gen_logits = self.decoder.greedy_decode(memory, max_len=self.max_len, sos_idx=self.sos_idx, eos_idx=self.eos_idx)
        return gen_logits, hidden_stub


# ----------------------------
# Minimal runnable self-test
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shapes / params
    B, C, H, W = 2, 3, 224, 224
    vocab_size = 5000
    in_shape = (B, C, H, W)
    out_shape = (vocab_size,)  # can be nested in your system; Net handles recursion

    prm = {
        "lr": 1e-4,
        "momentum": 0.9,
        "hidden_size": 768,
        "num_heads": 8,
        "num_decoding_layers": 1,
        "enc_layers": 1,
        "decoder_dropout": 0.1,
        "pad_idx": 0,
        "sos_idx": 1,
        "eos_idx": 2,
        "max_len": 16,
    }

    net = Net(in_shape, out_shape, prm, device).to(device)
    net.train_setup(prm)

    imgs = torch.randn(B, C, H, W, device=device)
    caps = torch.randint(0, vocab_size, (B, 12), device=device)
    caps[:, 0] = prm["sos_idx"]

    # Teacher forcing
    logits, _ = net(imgs, caps[:, :-1])
    print("Teacher forcing logits:", logits.shape)  # [B, T-1, V]

    # Greedy
    gen_logits, _ = net(imgs, captions=None)
    print("Generated logits:", gen_logits.shape)    # [B, T_gen, V]
