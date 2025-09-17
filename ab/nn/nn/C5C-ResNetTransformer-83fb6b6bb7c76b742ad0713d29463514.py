import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        s = self.avg(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 3, se_ratio: float = 0.5):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True)]
        else:
            hidden = in_ch
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ]
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)
        red = max(1, int(round(1.0 / se_ratio))) if se_ratio > 0 else 4
        self.se = SEBlock(out_ch, reduction=red) if se_ratio > 0 else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        if self.use_res:
            out = out + x
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 512, se_ratio: float = 0.5):
        super().__init__()
        c1, c2, c3 = 64, 128, hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.stage1 = InvertedResidual(c1, c1, stride=1, expand=3, se_ratio=se_ratio)
        self.stage2 = InvertedResidual(c1, c2, stride=2, expand=3, se_ratio=se_ratio)
        self.stage3 = InvertedResidual(c2, c3, stride=2, expand=3, se_ratio=se_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return x.unsqueeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerShim(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 1, dim_ff: int = 2048):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.num_layers = num_layers

    def init_zero_hidden(self, batch: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch, self.d_model, device=device)
        c0 = torch.zeros(self.num_layers, batch, self.d_model, device=device)
        return (h0, c0)

    def forward(self, inputs: torch.Tensor, hidden_state, features: torch.Tensor):
        x = self.embedding(inputs) * math.sqrt(self.d_model)
        x = self.pos(x)
        T = inputs.size(1)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=inputs.device), diagonal=1)
        out = self.dec(tgt=x, memory=features, tgt_mask=mask)
        logits = self.fc(out)
        return logits, hidden_state

    @torch.no_grad()
    def greedy_decode(self, features: torch.Tensor, max_len: int, sos: int = 1, eos: int = 2):
        B = features.size(0)
        ys = torch.full((B, 1), sos, dtype=torch.long, device=features.device)
        for _ in range(max_len):
            x = self.embedding(ys) * math.sqrt(self.d_model)
            x = self.pos(x)
            T = x.size(1)
            mask = torch.triu(torch.full((T, T), float('-inf'), device=ys.device), diagonal=1)
            out = self.dec(tgt=x, memory=features, tgt_mask=mask)
            next_logits = self.fc(out[:, -1, :])
            next_ids = next_logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_ids], dim=1)
            if (next_ids == eos).all():
                break
        return ys[:, 1:]


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1])
        vocab_size = int(out_shape[0])
        hidden_dim = int(prm.get('hidden_dim', 512))
        nhead = 8 if hidden_dim % 8 == 0 else 4
        self.encoder = Encoder(in_channels, hidden_dim=hidden_dim, se_ratio=0.5)
        self.rnn = TransformerShim(vocab_size=vocab_size, d_model=hidden_dim, nhead=nhead, num_layers=1, dim_ff=2048)
        self.vocab_size = vocab_size

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        beta1 = float(prm.get('momentum', 0.9))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm['lr']), betas=(beta1, 0.999), weight_decay=0.01)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            logits, _ = self(images, captions, None)
            tgt = (captions[:, 0, :] if captions.ndim == 3 else captions)[:, 1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images, captions=None, hidden_state=None):
        assert images.dim() == 4
        features = self.encoder(images)
        if captions is None:
            return self.rnn.greedy_decode(features, max_len=50)
        if captions.ndim == 3:
            captions = captions[:, 0, :]
        inputs = captions[:, :-1]
        assert inputs.dtype == torch.long
        if hidden_state is None:
            hidden_state = self.rnn.init_zero_hidden(images.size(0), images.device)
        logits, hidden_state = self.rnn(inputs, hidden_state, features)
        assert logits.dim() == 3 and logits.shape[1] == inputs.shape[1]
        return logits, hidden_state
