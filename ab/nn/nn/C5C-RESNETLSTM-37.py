import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, Any, Callable


# ----------------------------- helpers -----------------------------

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=groups, dilation=dilation)


# ------------------------ Squeeze-and-Excitation ------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation with robust ratio handling.

    ratio can be an int (reduction factor) or a float in (0,1] meaning 1/ratio.
    """
    def __init__(self, channels: int, ratio: float = 16.0) -> None:
        super().__init__()
        # interpret ratio
        if isinstance(ratio, float) and ratio <= 1.0:
            reduction = max(1, int(round(1.0 / max(1e-6, ratio))))
        else:
            reduction = max(1, int(round(ratio)))
        hidden = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# --------------------------- Bottleneck (SE) ---------------------------

class BottleneckSEBlock(nn.Module):
    """ResNet bottleneck with SE."""
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.se = SEBlock(width, ratio=se_ratio)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# --------------------------- Encoder (SE-ResNet) ---------------------------

class EncoderSEResNet(nn.Module):
    """Small SE-ResNet encoder that outputs region features [B, N, D]."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 768, se_ratio: float = 0.25) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Two stages keep it lightweight; spatial downsample by /4 more (total /8)
        self.layer1 = self._make_layer(planes=64, blocks=2, stride=1, se_ratio=se_ratio)
        self.layer2 = self._make_layer(planes=128, blocks=2, stride=2, se_ratio=se_ratio)

        # 1x1 projection to embed_dim
        self.proj = nn.Conv2d(128 * BottleneckSEBlock.expansion, embed_dim, kernel_size=1, bias=False)

    def _make_layer(self, planes: int, blocks: int, stride: int, se_ratio: float) -> nn.Sequential:
        downsample = None
        out_channels = planes * BottleneckSEBlock.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BottleneckSEBlock(self.inplanes, planes, stride=stride, downsample=downsample, se_ratio=se_ratio)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(BottleneckSEBlock(self.inplanes, planes, stride=1, downsample=None, se_ratio=se_ratio))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)                     # [B, C2, H', W']
        x = self.proj(x)                       # [B, D,  H', W']
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, D)  # [B, N, D]
        return x


# ------------------------ Transformer Caption Decoder ------------------------

class TransformerCaptionDecoder(nn.Module):
    """Transformer decoder over token embeddings; memory is encoder regions [B, N, D]."""
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

        self.tok_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = nn.Embedding(1024, self.embed_dim)

        layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.out_proj = nn.Linear(self.embed_dim, self.vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # True = mask; shape [T,T]
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        memory: torch.Tensor,                     # [B, N, D]
        captions_in: Optional[torch.Tensor],      # [B, T_in] or None (greedy)
        max_len: int,
        sos_token: int,
        eos_token: int,
    ) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device

        if captions_in is not None:
            T = captions_in.size(1)
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            tgt = self.tok_embed(captions_in) + self.pos_embed(pos)        # [B,T,D]
            tgt_mask = self._causal_mask(T, device)                        # [T,T]
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # [B,T,D]
            logits = self.out_proj(dec)                                    # [B,T,V]
            return logits

        # Greedy decoding
        ys = torch.full((B, 1), sos_token, dtype=torch.long, device=device)
        for _ in range(max_len):
            T = ys.size(1)
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            tgt = self.tok_embed(ys) + self.pos_embed(pos)
            tgt_mask = self._causal_mask(T, device)
            dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            logit = self.out_proj(dec[:, -1, :])                           # [B,V]
            next_tok = logit.argmax(dim=-1, keepdim=True)                  # [B,1]
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_token).all():
                break
        return ys  # [B, <=max_len+1]


# ------------------------------------ Net ------------------------------------

class Net(nn.Module):
    """Main Image Captioning Model (SE-ResNet encoder + Transformer decoder)."""
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.device = device

        # infer vocab_size from nested out_shape (robust)
        def _flatten_ints(x):
            if isinstance(x, (tuple, list)):
                for xi in x:
                    yield from _flatten_ints(xi)
            else:
                try:
                    yield int(x)
                except Exception:
                    pass

        ints = list(_flatten_ints(out_shape))
        if not ints:
            raise ValueError("out_shape must encode vocab size (e.g., ((V,),) or similar).")
        self.vocab_size = max(2, int(ints[0]))

        # config
        self.hidden_size = int(prm.get("hidden_size", 768))  # ≥ 640
        self.num_heads = int(prm.get("num_heads", 8))
        self.se_ratio = float(prm.get("se_ratio", 0.25))
        self.dropout_p = float(prm.get("dropout", 0.1))
        self.max_len = int(prm.get("max_len", 20))
        self.sos_token = int(prm.get("sos_token", 1))
        self.eos_token = int(prm.get("eos_token", 2))
        self.pad_token = int(prm.get("pad_token", 0))

        # infer in_channels from in_shape (supports (C,H,W) or (N,C,H,W))
        def _infer_in_channels(shape) -> int:
            if isinstance(shape, (tuple, list)):
                if len(shape) == 3:
                    return int(shape[0])
                if len(shape) >= 4:
                    return int(shape[1])
            return 3

        in_channels = _infer_in_channels(in_shape)

        # modules
        self.encoder = EncoderSEResNet(in_channels=in_channels, embed_dim=self.hidden_size, se_ratio=self.se_ratio)
        self.decoder = TransformerCaptionDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=int(prm.get("num_layers", 4)),
            dropout=self.dropout_p,
        )

        # training state
        self.criteria = None
        self.criterion = None
        self.optimizer = None

        self.to(self.device)

    # -------------------------- training plumbing --------------------------

    def train_setup(self, prm: Dict[str, Any]) -> None:
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token).to(self.device)
        self.criteria = (self.criterion,)
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=float(prm.get("lr", 1e-3)),
            betas=(float(prm.get("momentum", 0.9)), 0.999),
        )

    def learn(self, train_data) -> None:
        """Minimal example loop. Expects batches of (images, captions[B,T])."""
        if self.optimizer is None or self.criterion is None:
            self.train_setup({})

        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            # teacher forcing: predict next tokens
            inp = captions[:, :-1]          # [B,T-1]
            tgt = captions[:, 1:]           # [B,T-1]

            memory = self.encoder(images)   # [B,N,D]
            logits = self.decoder(memory, inp, self.max_len, self.sos_token, self.eos_token)  # [B,T-1,V]

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    # ------------------------------- forward -------------------------------

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # unused for Transformer
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        If captions is provided (teacher forcing):
            returns (logits[B,T-1,V], None)
        Else (greedy inference):
            returns (tokens[B,<=max_len+1], None)
        """
        assert images.dim() == 4, f"images must be [B,C,H,W], got {tuple(images.shape)}"
        memory = self.encoder(images.to(self.device))  # [B,N,D]

        if captions is not None:
            captions = captions.to(self.device)
            assert captions.dim() == 2, "captions must be [B,T]"
            inp = captions[:, :-1]
            logits = self.decoder(memory, inp, self.max_len, self.sos_token, self.eos_token)
            return logits, None
        else:
            tokens = self.decoder(memory, None, self.max_len, self.sos_token, self.eos_token)
            return tokens, None


def supported_hyperparameters():
    return {'lr', 'momentum'}
