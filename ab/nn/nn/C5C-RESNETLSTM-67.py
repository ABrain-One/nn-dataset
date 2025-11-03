import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Iterable


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):
    """
    CNN encoder + Transformer decoder image captioning model.

    API preserved:
      - __init__(in_shape, out_shape, prm, device)
      - train_setup(prm)
      - learn(train_data)
      - forward(images, captions=None, hidden_state=None)
      - init_zero_hidden(batch, device)
    """
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device

        # ---- Hyperparameters / config ----
        self.hidden_size = int(prm.get('hidden_size', 768))            # >= 640 as in prior specs
        self.max_len = int(prm.get('max_len', 20))
        self.nhead = int(prm.get('nhead', 8))
        self.num_layers = int(prm.get('num_layers', 1))
        self.dropout = float(prm.get('dropout', 0.1))
        self.pad_idx = int(prm.get('pad_idx', 0))
        self.sos_idx = int(prm.get('sos_idx', 1))
        self.eos_idx = int(prm.get('eos_idx', 2))

        # Infer channels from in_shape that may be (B,C,H,W) or (C,H,W)
        if len(in_shape) == 4:
            in_channels = int(in_shape[1])
        elif len(in_shape) == 3:
            in_channels = int(in_shape[0])
        else:
            raise ValueError(f"in_shape should be (C,H,W) or (B,C,H,W), got {in_shape}")

        # Robust vocab size extraction (handles nested tuples like out_shape[0][0][0])
        self.vocab_size = self._first_int(out_shape)
        if self.vocab_size <= 0:
            raise ValueError(f"Invalid out_shape {out_shape}: could not infer a positive vocab size")

        # ---- Encoder (your original conv stack + GAP + projection) ----
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B,512,1,1]
            nn.Flatten(),                  # -> [B,512]
        )
        self.encoder_project = nn.Linear(512, self.hidden_size)  # -> [B,hidden]

        # ---- Decoder (Transformer) ----
        # Use batch_first=True to simplify shapes: (B, T, E)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        self.fc_out = nn.Linear(self.hidden_size, self.vocab_size)

        # Init
        self.apply(self._initialize_weights)
        self.to(self.device)

        # Will be set in train_setup
        self.criteria: Tuple[nn.Module, ...] = ()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.grad_clip = float(prm.get('grad_clip', 3.0))

    # ---------- helpers ----------
    @staticmethod
    def _first_int(x) -> int:
        """Extract first int from possibly nested tuples/lists."""
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return Net._first_int(x[0])
        raise ValueError(f"Cannot infer integer from {x}")

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        # Bool mask: True where positions should be masked (upper triangle)
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    # ---------- training API ----------
    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        lr = float(prm.get('lr', 1e-4))
        beta1 = float(prm.get('momentum', 0.9))
        self.criteria = (nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=(beta1, 0.999)
        )

    def learn(self, train_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        train_data: iterable of (images, captions)
          images   : [B, C, H, W]
          captions : [B, T] (token ids with SOS at pos 0; PAD = self.pad_idx)
        """
        if self.optimizer is None or not self.criteria:
            raise RuntimeError("Call train_setup(prm) before learn().")

        self.train()
        for images, captions in train_data:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            # Teacher forcing: predict next tokens
            inp = captions[:, :-1]   # [B, T-1]
            tgt = captions[:, 1:]    # [B, T-1]

            logits, _ = self.forward(images, inp)  # [B, T-1, V]
            loss = self.criteria[0](logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()

    # ---------- forward ----------
    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        If captions is provided (teacher forcing):
          images:   [B,C,H,W]
          captions: [B,T] token ids (no need to include SOS here—pass the input sequence directly)
          returns:  (logits [B,T,V], hidden_state_stub)

        If captions is None (greedy decoding):
          returns:  (logits [B,T_gen,V], hidden_state_stub)
        """
        B = images.size(0)

        # ----- Encoder -----
        features = self.encoder(images)              # [B, 512]
        memory = self.encoder_project(features)      # [B, H]
        memory = memory[:, None, :]                  # [B, 1, H]   (sequence len 1 for image-global token)

        # A stub "hidden state" to match the expected return signature
        hidden_stub = self.init_zero_hidden(B, images.device)

        # ----- Teacher forcing path -----
        if captions is not None:
            # Embed targets and apply causal mask
            tgt_emb = self.embedding(captions)       # [B, T, H]
            T = tgt_emb.size(1)
            tgt_mask = self._generate_square_subsequent_mask(T, images.device)  # [T,T]

            # Transformer decode
            dec_out = self.transformer(
                tgt=tgt_emb,                # [B,T,H]
                memory=memory,              # [B,1,H]
                tgt_mask=tgt_mask,          # [T,T] causal
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            )                               # [B,T,H]

            logits = self.fc_out(dec_out)   # [B,T,V]
            return logits, hidden_stub

        # ----- Greedy decoding path -----
        generated_logits = []
        tokens = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)  # [B,1]

        for _ in range(self.max_len):
            tgt_emb = self.embedding(tokens)          # [B,t,H]
            t = tgt_emb.size(1)
            tgt_mask = self._generate_square_subsequent_mask(t, images.device)

            dec_out = self.transformer(
                tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask
            )                                          # [B,t,H]

            step_logits = self.fc_out(dec_out[:, -1:, :])  # [B,1,V] last step
            generated_logits.append(step_logits)

            next_tok = step_logits.argmax(-1)          # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)

            if (next_tok.squeeze(1) == self.eos_idx).all():
                break

        logits = torch.cat(generated_logits, dim=1)    # [B,T_gen,V]
        return logits, hidden_stub

    def init_zero_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used by Transformer, kept for API compatibility with prior models
        h0 = torch.zeros(1, batch, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch, self.hidden_size, device=device)
        return h0, c0


# ---------------------- quick self-test ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shapes consistent with prior tasks
    B, C, H, W = 2, 3, 224, 224
    vocab_size = 1000
    in_shape = (B, C, H, W)
    out_shape = (vocab_size,)
    prm = {"lr": 1e-4, "momentum": 0.9, "hidden_size": 768, "nhead": 8, "num_layers": 1, "dropout": 0.1}

    model = Net(in_shape, out_shape, prm, device)
    model.train_setup(prm)

    images = torch.randn(B, C, H, W, device=device)
    captions = torch.randint(0, vocab_size, (B, 12), device=device)
    captions[:, 0] = 1  # SOS
    captions[:, -1] = 2 # EOS

    # Teacher forcing
    logits, _ = model(images, captions[:, :-1])
    print("Teacher forcing logits:", logits.shape)  # [B, T-1, V]

    # Greedy generation
    gen_logits, _ = model(images, captions=None)
    print("Generated logits:", gen_logits.shape)    # [B, T_gen, V]
