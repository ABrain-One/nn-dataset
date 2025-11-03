import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}


class BagNetEncoder(nn.Module):
    """
    Produces a sequence of visual tokens for the Transformer decoder.
    Output: [B, L, hidden] where L = 7*7 = 49 tokens.
    """
    def __init__(self, in_channels=3, hidden_size=768):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)
        self.layer5 = self._make_layer(512, 512, 2)
        self.layer6 = self._make_layer(512, hidden_size, 2)

        # Pool to a fixed 7x7 grid, then flatten to tokens.
        self.token_pool = nn.AdaptiveAvgPool2d((7, 7))

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu0(self.bn0(self.initial(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)               # [B, hidden, H, W]
        x = self.token_pool(x)           # [B, hidden, 7, 7]
        x = x.flatten(2).transpose(1, 2) # [B, 49, hidden]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = nn.Embedding(512, hidden_size)

        layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # For API compatibility (Transformers don't use RNN hidden state)
        self.init_zero_hidden = lambda batch_size, device: None

    def _subsequent_mask(self, T: int, device: torch.device):
        # (T, T) mask with -inf above diagonal to prevent attending to future positions
        m = torch.full((T, T), float('-inf'), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, captions, memory, hidden_state=None):
        """
        captions: [B, T] token ids
        memory:   [B, L, hidden] encoder tokens
        """
        B, T = captions.shape
        device = captions.device

        tok = self.embedding(captions)  # [B, T, H]
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tok = tok + self.pos_encoder(pos_ids)

        tgt_mask = self._subsequent_mask(T, device)  # [T, T]
        out = self.transformer_decoder(tgt=tok, memory=memory, tgt_mask=tgt_mask)  # [B, T, H]
        logits = self.fc_out(out)  # [B, T, V]
        return logits, hidden_state


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        # ---- API aliases (auto-injected) ----
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3

        # Try common out_shape patterns
        try:
            self.vocab_size = int(out_shape[0][0])
        except Exception:
            try:
                self.vocab_size = int(out_shape[0])
            except Exception:
                self.vocab_size = int(out_shape)

        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size

        # Encoder / Decoder
        self.encoder = BagNetEncoder(in_channels=self.in_channels, hidden_size=768)
        self.decoder = TransformerDecoder(hidden_size=768, num_heads=8, vocab_size=self.vocab_size)

        # Expose RNN-style helper for compatibility
        self.init_zero_hidden = self.decoder.init_zero_hidden

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', 1e-3)),
            betas=(float(prm.get('momentum', 0.9)), 0.999),
        )

    def learn(self, train_data):
        # placeholder hook — training loop is dataset/loader-specific
        pass

    def forward(self, images, captions=None, hidden_state=None):
        """
        Training:  images [B, C, H, W], captions [B, T] -> logits [B, T, V]
        Inference: images [B, C, H, W] -> greedy tokens [B, T]
        """
        images = images.to(self.device)
        memory = self.encoder(images)  # [B, L, H]

        if captions is not None:
            if captions.dim() == 3:  # e.g., [B, 1, T]
                captions = captions.squeeze(1)
            captions = captions.to(self.device).long()
            logits, hidden_state = self.decoder(captions, memory, hidden_state)
            return logits, hidden_state

        # Inference: greedy decode
        B = images.size(0)
        max_len = int(self.prm.get('max_len', 20))
        sos_id = int(self.prm.get('sos', 0))
        eos_id = int(self.prm.get('eos', 1))

        seq = torch.full((B, 1), sos_id, device=self.device, dtype=torch.long)
        for _ in range(max_len):
            logits, _ = self.decoder(seq, memory, None)   # [B, T, V]
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # [B, 1]
            seq = torch.cat([seq, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return seq, None
