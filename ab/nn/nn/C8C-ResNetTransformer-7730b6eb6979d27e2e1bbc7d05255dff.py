import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = self.pool(x)
        return x * self.sigmoid(edge)


class CABlock(nn.Module):
    def __init__(self, channel, reduction=4, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channel, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
        self.conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        out = x.clone()
        out = self.conv(out)
        out = self.channel_att(out) * out + out
        out = self.spatial_att(out)
        return out


class CBAM(nn.Module):
    def __init__(self, channel, reduction=4, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channel, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.channel(x) * x + self.spatial(x) * x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand=4.0, kernel_size=5, stride=1, se_ratio=4.0, drop_rate=0.1, index=0):
        super().__init__()
        self.has_se = (out_channels != in_channels)
        self.depth_multiplier = expand
        self.pointwise_conv1 = nn.Conv2d(in_channels, int(in_channels * self.depth_multiplier), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(in_channels * self.depth_multiplier))
        self.act1 = nn.GELU()
        self.depth_conv = nn.Conv2d(int(in_channels * self.depth_multiplier), int(in_channels * self.depth_multiplier), kernel_size, padding=kernel_size//2, groups=int(in_channels * self.depth_multiplier), stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(int(in_channels * self.depth_multiplier))
        self.act2 = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(int(in_channels * self.depth_multiplier), out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels) if self.has_se else None
        mid_channels = max(1, int(out_channels // (int(se_ratio) if isinstance(se_ratio, (int, float)) and se_ratio >= 1 else 1)))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(int(in_channels * self.depth_multiplier), mid_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels, bias=True),
            nn.Sigmoid(),
            nn.Unflatten(1, (out_channels, 1, 1))
        )
        
    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.depth_conv.kernel_size == (1, 1):
            shortcut = x
        else:
            x = self.depth_conv(x)
            x = self.bn2(x)
            x = self.act2(x)
            shortcut = None
        x = self.pointwise_conv2(x)
        if self.has_se and shortcut is not None and x.shape[1] == shortcut.shape[1]:
            x = self.se(x + shortcut)
        return x


class ScConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, groups=64, reduction=4, deploy=False):
        super().__init__()
        self.identity_connection = in_channels == out_channels and stride == 1
        self.padding = kernel_size // 2
        self.empty = False if not deploy else False
        self.pointwise1 = nn.Conv2d(in_channels, out_channels, 1)
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels) if self.identity_connection else None
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_features=in_channels, out_features=in_channels//16)
        ) if not self.empty else None

    def forward(self, x):
        identity = x
        y = self.pointwise1(x)
        if not self.empty:
            y = self.depth_conv(y)
            y = self.bn2(y)
            y = self.act(y)
            y = self.pointwise2(y)
        if self.bn3 is not None:
            y = self.bn3(y)
        if self.identity_connection and y.shape == identity.shape:
            y = identity + y
        return y


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_from_feat = nn.Linear(d_model, hidden_size)

    def init_zero_hidden(self, batch, device):
        h0 = torch.zeros(1, batch, self.gru.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(self, inputs, hidden_state=None, features=None):
        emb = self.embedding(inputs)
        if hidden_state is None:
            if features is not None:
                h = self.init_from_feat(features).unsqueeze(0)
            else:
                h = torch.zeros(1, inputs.size(0), self.gru.hidden_size, device=inputs.device)
        else:
            h = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
        out, h = self.gru(emb, h)
        logits = self.fc(out)
        return logits, (h, torch.zeros_like(h))

    def greedy_decode(self, features, max_len=20, start_id=1, end_id=2):
        B = features.size(0)
        device = features.device
        h = self.init_from_feat(features).unsqueeze(0)
        seq = torch.full((B, 1), start_id, dtype=torch.long, device=device)
        tokens = []
        for _ in range(max_len):
            emb = self.embedding(seq[:, -1:])
            out, h = self.gru(emb, h)
            next_logits = self.fc(out[:, -1, :])
            next_ids = next_logits.argmax(-1, keepdim=True)
            tokens.append(next_ids)
            seq = torch.cat([seq, next_ids], dim=1)
            if (next_ids == end_id).all():
                break
        if len(tokens) == 0:
            return torch.zeros((B, 0), dtype=torch.long, device=device)
        return torch.cat(tokens, dim=1)


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        vocab_size = int(out_shape[0])
        self.embed_dim = 768
        self.num_heads = 8
        self.num_layers = 6
        self.dropout_rate = 0.1
        self.prj = prm.get('prj', {})
        self.dropout = getattr(prm, 'dropout', 0.1)
        self.attention_dropout = getattr(prm, 'attention_dropout', 0.1)
        self.use_checkpointing = getattr(prm, 'use_checkpointing', False)
        self.use_mem_efficient = getattr(prm, 'use_mem_efficient', True)
        self.projection = nn.Conv2d(3, self.embed_dim, 3, bias=False)
        self.tokenization = nn.AdaptiveAvgPool2d(1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.hybrid_encoder = nn.Sequential(
            CBAM(64),
            CBAM(64),
            ScConv(64, 64),
            MBConvBlock(64, 64)
        )
        self.transformer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dropout=self.dropout_rate, batch_first=True)
        self.fc = nn.Linear(self.embed_dim, vocab_size)
        self.rnn = Decoder(vocab_size=vocab_size, d_model=self.embed_dim, hidden_size=640)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def forward(self, images, captions=None, hidden_state=None):
        assert images.dim() == 4
        feats = self.tokenization(self.projection(images)).flatten(1)
        if captions is not None:
            if captions.ndim == 3:
                captions = captions[:, 0, :]
            inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(inputs, hidden_state, features=feats)
            assert logits.shape[1] == inputs.shape[1]
            return logits, hidden_state
        preds = self.rnn.greedy_decode(feats, max_len=20)
        return preds

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            logits, _ = self.forward(images, captions, None)
            tgt = (captions[:, 0, :] if captions.ndim == 3 else captions)[:, 1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
