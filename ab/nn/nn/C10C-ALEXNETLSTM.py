import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

# --- AlexNet weights (safe fallback for older torchvision) ---
try:
    from torchvision.models import AlexNet_Weights
    ALEXNET_W = AlexNet_Weights.IMAGENET1K_V1
except Exception:
    ALEXNET_W = None  # fallback: use pretrained=True on older torchvision

# Optional: dynamic special-token ids (if dataset loader sets them)
try:
    from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
except Exception:
    GLOBAL_CAPTION_VOCAB = {}

def supported_hyperparameters():
    # repo's train.py consumes lr/momentum/dropout via -p JSON or optuna ranges
    return {'lr', 'momentum', 'dropout'}

# ---------- helpers ----------
def _special_ids(vocab: dict, vocab_size: int):
    def hit(keys, default):
        for k in keys:
            if k in vocab:
                return int(vocab[k])
        return max(0, min(default, vocab_size - 1))
    pad = hit(['<PAD>', '<pad>', '<pad_token>', '<blank>', '<null>'], 0)
    bos = hit(['<BOS>', '<bos>', '<s>', '<start>', '<SOS>', '<sos>'], 1)
    eos = hit(['<EOS>', '<eos>', '</s>', '<end>', '<EOS_TOKEN>'], 2)
    return pad, bos, eos

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, L, D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return self.drop(x + self.pe[:, :L, :])

# ---------- encoder (AlexNet features) ----------
class CNNEncoder(nn.Module):
    """
    AlexNet conv feature extractor.
    Output: (B, 256, h, w)
    """
    def __init__(self, in_ch: int, feat_ch: int = 256):  # feat_ch kept for signature compatibility
        super().__init__()
        # torchvision AlexNet expects 3-channel RGB; repo transforms handle resize/normalize
        if ALEXNET_W is None:
            self.backbone = tv.alexnet(pretrained=True).features
        else:
            self.backbone = tv.alexnet(weights=ALEXNET_W).features

    def forward(self, x):
        return self.backbone(x)  # (B, 256, h, w)

# ---------- decoder (Transformer) ----------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos   = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True,
            dropout=dropout,
            activation='gelu',
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc  = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """
        Boolean causal mask (True = blocked) to match key_padding_mask dtype,
        avoiding 'mismatched key_padding_mask and attn_mask' warning.
        Shape: (L, L)
        """
        m = torch.ones((L, L), dtype=torch.bool, device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: (B, T), memory: (B, S, D)
        returns logits: (B, T, V)
        """
        B, T = tgt_tokens.shape
        x = self.embed(tgt_tokens)                    # (B,T,D)
        x = self.pos(x)
        tgt_mask = self._causal_mask(T, x.device)     # (T,T) bool
        tgt_kpm  = (tgt_tokens == self.pad_idx)       # (B,T) bool
        y = self.dec(tgt=x, memory=memory,
                     tgt_mask=tgt_mask,
                     tgt_key_padding_mask=tgt_kpm)
        return self.fc(y)                             # (B,T,V)

# ---------- full model ----------
class Net(nn.Module):
    """
    Returns a Tensor from forward() (never a tuple), so metrics like BLEU can call .dim().
    API:
      - __init__(in_shape, out_shape, prm, device)
      - forward(images, captions=None) -> logits Tensor
      - train_setup(prm)
      - learn(train_data)
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = int(in_shape[1])
        self.vocab_size  = int(out_shape[0])

        # Hyperparams
        self.dropout_p = float(prm.get('dropout', 0.1))
        self.max_len   = int(prm.get('max_len', 20))

        # Special tokens
        self.pad_idx, self.bos_id, self.eos_id = _special_ids(GLOBAL_CAPTION_VOCAB or {}, self.vocab_size)

        # Encoder -> sequence of d_model features
        d_model = 512
        enc_feat_ch = 256   # AlexNet conv5 output channels
        self.encoder = CNNEncoder(self.in_channels, feat_ch=enc_feat_ch)
        self.enc_proj = nn.Linear(enc_feat_ch, d_model)
        self.enc_pos  = PositionalEncoding(d_model, dropout=self.dropout_p)
        self.enc_drop = nn.Dropout(self.dropout_p)

        # Transformer decoder
        self.decoder = TransformerDecoder(
            self.vocab_size,
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dropout=self.dropout_p,
            pad_idx=self.pad_idx,
        )

        # Training attrs init in train_setup
        self.criteria = None
        self.optimizer = None
        self.scaler = None

    # -- encoder helper --
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        f = self.encoder(images)                     # (B,C,h,w) with C=256
        B, C, h, w = f.shape
        seq = f.view(B, C, h*w).permute(0, 2, 1)    # (B,S,C) where S=h*w
        seq = self.enc_proj(seq)                    # (B,S,D)
        seq = self.enc_pos(seq)
        seq = self.enc_drop(seq)
        return seq                                   # (B,S,D)

    # -- forward --
    def forward(self, images, captions=None):
        """
        Training (teacher forcing):
            inputs = captions[:, :-1] -> logits over positions 1..T-1
            returns logits: (B, T-1, V)
        Inference (captions=None):
            greedy decode up to max_len
            returns logits: (B, L, V) of generated steps
        """
        assert images.dim() == 4, "images must be (B,C,H,W)"
        memory = self._encode(images)  # (B,S,D)

        if captions is not None:
            if captions.ndim == 3:
                captions = captions[:, 0, :]        # (B,T)
            inputs = captions[:, :-1]               # (B,T-1)
            logits = self.decoder(inputs, memory)   # (B,T-1,V)
            return logits                           # Tensor ONLY

        # Inference: greedy
        B = images.size(0)
        device = images.device
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        steps = []
        for _ in range(self.max_len):
            step_logits = self.decoder(cur, memory)[:, -1:, :]  # (B,1,V)
            steps.append(step_logits)
            next_tok = step_logits.argmax(dim=-1)               # (B,1)
            cur = torch.cat([cur, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.eos_id).all():
                break
        logits = torch.cat(steps, dim=1) if steps else torch.zeros((B, 0, self.vocab_size), device=device)
        return logits                                           # Tensor ONLY

    # -- training setup --
    def train_setup(self, prm):
        self.to(self.device)
        # Loss: ignore PAD; label smoothing helps BLEU
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=0.1).to(self.device),)
        # Map "momentum" → AdamW beta1 to keep CLI semantics
        beta1 = float(prm.get('momentum', 0.9))
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm['lr']),
            betas=(beta1, 0.999),
            weight_decay=1e-4,
        )
        # AMP (PyTorch 2.x API)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

    # -- one epoch training loop --
    def learn(self, train_data):
        """
        Expects batches like (images, captions, *rest).
        """
        assert self.criteria and self.optimizer is not None and self.scaler is not None, "Call train_setup(prm) first."
        self.train()
        amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        for batch in train_data:
            if isinstance(batch, (list, tuple)):
                images, captions = batch[0], batch[1]
            else:
                images, captions = batch
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=(self.device.type == 'cuda')):
                if captions.ndim == 3:
                    captions = captions[:, 0, :]
                logits = self.forward(images, captions)          # (B,T-1,V)
                targets = captions[:, 1:]                        # (B,T-1)
                loss = self.criteria[0](logits.reshape(-1,logits.size(-1)),
                                        targets.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
