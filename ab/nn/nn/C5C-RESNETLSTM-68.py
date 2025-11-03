import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable


def supported_hyperparameters():
    return {'lr', 'momentum'}


# ---------------------------
# Utility / building blocks
# ---------------------------
class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class BBBConv2d(nn.Module):
    """
    Bayesian-ish wrapper around Conv2d.
    For executability and simplicity, priors are ignored (no-ops).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        priors=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.priors = priors  # kept for API compatibility

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        # If you had priors/sampling logic, you'd apply it here.
        return self.conv(x)


class BBBLinear(nn.Module):
    """
    Bayesian-ish wrapper around Linear.
    For executability and simplicity, priors are ignored (no-ops).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, priors=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.priors = priors

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        # If you had priors/sampling logic, you'd apply it here.
        return self.linear(x)


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.drop1(attn_out)
        # MLP + residual
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


class Encoder(nn.Module):
    """
    Lightweight Transformer-style encoder that works on a short sequence (here we use 1 token: the image embedding).
    """
    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.pos_embedding = nn.Embedding(512, hidden_dim)  # max 512 positions; we only need a few
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, mlp_dim, dropout, attention_dropout)
        for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H]
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embedding(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class DecoderBlock(nn.Module):
    """
    Simple cross-attention + MLP block (kept to preserve original structure name).
    Not directly used because we rely on nn.TransformerDecoder for correctness and simplicity.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        if memory is not None:
            attn_out, _ = self.cross_attn(self.ln1(x), memory, memory, need_weights=False)
            x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# ---------------------------
# Image Captioning Net
# ---------------------------
class Net(nn.Module):
    """
    Executable image captioning model with:
      - CNN encoder -> project to hidden token
      - Transformer encoder (lightweight)
      - Transformer decoder (PyTorch)
    Preserves names/classes from the provided structure (FlattenLayer, BBBLinear, Encoder, DecoderBlock, Net).
    """
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm

        # ---- Config ----
        self.hidden_dim = int(prm.get('hidden_dim', 768))
        self.num_heads = int(prm.get('num_heads', 8))
        self.num_enc_layers = int(prm.get('enc_layers', 1))
        self.num_dec_layers = int(prm.get('dec_layers', 1))
        self.mlp_dim = int(prm.get('mlp_dim', 4 * self.hidden_dim))
        self.dropout = float(prm.get('dropout', 0.1))
        self.attn_dropout = float(prm.get('attention_dropout', 0.1))
        self.pad_idx = int(prm.get('pad_idx', 0))
        self.sos_idx = int(prm.get('sos_idx', 1))
        self.eos_idx = int(prm.get('eos_idx', 2))
        self.max_len = int(prm.get('max_len', 20))

        # Infer channels from in_shape that may be (B,C,H,W) or (C,H,W)
        if len(in_shape) == 4:
            in_channels = int(in_shape[1])
        elif len(in_shape) == 3:
            in_channels = int(in_shape[0])
        else:
            raise ValueError(f"in_shape should be (C,H,W) or (B,C,H,W), got {in_shape}")

        # Robust vocab size extraction from possibly nested out_shape
        self.vocab_size = self._first_int(out_shape)
        if self.vocab_size <= 0:
            raise ValueError(f"Invalid out_shape {out_shape}: could not infer a positive vocab size")

        # ---- CNN encoder -> [B, hidden_dim] ----
        self.cnn = nn.Sequential(
            BBBConv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            BBBConv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            BBBConv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            BBBConv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            BBBConv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            FlattenLayer(),
        )
        self.img_proj = BBBLinear(256, self.hidden_dim)

        # ---- Transformer Encoder on the single image token ----
        self.encoder = Encoder(
            num_layers=self.num_enc_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            attention_dropout=self.attn_dropout,
        )

        # ---- Transformer Decoder ----
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_idx)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.mlp_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.num_dec_layers)
        self.output_proj = BBBLinear(self.hidden_dim, self.vocab_size)

        # Init + device
        self.apply(self._init_weights)
        self.to(self.device)

        # training artifacts
        self.criteria = ()
        self.optimizer = None
        self.grad_clip = float(prm.get('grad_clip', 3.0))

    # --------- helper methods ----------
    @staticmethod
    def _first_int(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Cannot infer integer from {x}")

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear, BBBLinear)):
            try:
                nn.init.kaiming_normal_(m.weight if isinstance(m, nn.Linear) else m.linear.weight, nonlinearity='relu')
            except Exception:
                # For BBBLinear we store layer inside .linear
                if isinstance(m, BBBLinear):
                    nn.init.kaiming_normal_(m.linear.weight, nonlinearity='relu')
            if isinstance(m, (nn.Linear, BBBLinear)):
                if isinstance(m, BBBLinear):
                    if m.linear.bias is not None:
                        nn.init.constant_(m.linear.bias, 0)
                else:
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # Bool mask: True in upper triangle (masked positions)
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Kept for API compatibility with earlier RNN-based variants
        h0 = torch.zeros(1, batch, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h0, c0

    # --------- training API ----------
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        """
        train_data: iterable of (images, captions)
          images   : [B, C, H, W]
          captions : [B, T] (token ids; include SOS at pos 0; PAD = self.pad_idx)
        """
        if not self.criteria or self.optimizer is None:
            raise RuntimeError("Call train_setup(prm) before learn().")

        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            # Teacher forcing: input is captions[:, :-1], target is captions[:, 1:]
            inp = captions[:, :-1]
            tgt = captions[:, 1:]

            logits, _ = self.forward(images, inp)  # [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

            yield float(loss.detach().cpu())

    # --------- forward ----------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        If captions is provided (teacher forcing):
          images:   [B,C,H,W]
          captions: [B,T] token ids (no need to include EOS in input)
          returns:  (logits [B,T,V], hidden_state_stub)

        If captions is None (greedy decoding):
          returns:  (logits [B,T_gen,V], hidden_state_stub)
        """
        B = images.size(0)

        # ----- CNN encoder -> image token -----
        img_feat = self.cnn(images)             # [B, 256]
        img_tok = self.img_proj(img_feat)       # [B, hidden_dim]
        memory = img_tok[:, None, :]            # [B, 1, hidden_dim]

        # Transformer encoder over the single token (kept for structure)
        memory = self.encoder(memory)           # [B, 1, hidden_dim]

        # Stub hidden state for API compatibility
        hidden_stub = self.init_zero_hidden(B, images.device)

        # ----- Teacher forcing -----
        if captions is not None:
            tgt_emb = self.embedding(captions)      # [B, T, H]
            T = tgt_emb.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, images.device)

            dec_out = self.decoder(
                tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
                memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None
            )                                       # [B, T, H]
            logits = self.output_proj(dec_out)      # [B, T, V]
            return logits, hidden_stub

        # ----- Greedy decoding -----
        generated_logits = []
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)  # [B,1]

        for _ in range(self.max_len):
            tgt_emb = self.embedding(tokens)         # [B,t,H]
            t = tgt_emb.size(1)
            tgt_mask = self._generate_square_subsequent_mask(t, images.device)

            dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)  # [B,t,H]
            step_logits = self.output_proj(dec_out[:, -1:, :])                      # [B,1,V]
            generated_logits.append(step_logits)

            next_tok = step_logits.argmax(-1)  # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)

            if (next_tok.squeeze(1) == self.eos_idx).all():
                break

        logits = torch.cat(generated_logits, dim=1)  # [B,T_gen,V]
        return logits, hidden_stub


# ---------------------- quick self-test ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example shapes
    B, C, H, W = 2, 3, 224, 224
    vocab_size = 5000
    in_shape = (B, C, H, W)
    out_shape = (vocab_size,)  # may be nested in your pipeline; Net handles that robustly
    prm = {
        "lr": 1e-4,
        "momentum": 0.9,
        "hidden_dim": 768,
        "num_heads": 8,
        "enc_layers": 1,
        "dec_layers": 1,
        "mlp_dim": 3072,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "pad_idx": 0,
        "sos_idx": 1,
        "eos_idx": 2,
        "max_len": 20,
    }

    model = Net(in_shape, out_shape, prm, device).to(device)
    model.train_setup(prm)

    images = torch.randn(B, C, H, W, device=device)
    captions = torch.randint(0, vocab_size, (B, 12), device=device)
    captions[:, 0] = prm["sos_idx"]
    captions[:, -1] = prm["eos_idx"]

    # Teacher forcing
    logits, _ = model(images, captions[:, :-1])
    print("Teacher forcing logits:", logits.shape)  # [B, T-1, V]

    # Greedy generation
    gen_logits, _ = model(images, captions=None)
    print("Generated logits:", gen_logits.shape)    # [B, T_gen, V]
