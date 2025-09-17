import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr','momentum'}

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc2(F.relu(self.fc1(y), inplace=True)).view(b, c, 1, 1)
        return x * torch.sigmoid(y)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return F.relu(self.bn(x), inplace=True)

class YourEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super().__init__()
        h2 = hidden_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, h2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(h2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(h2, hidden_dim),
            SEBlock(hidden_dim),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.stem(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size, feature_dim):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(feature_dim, hidden_size, bias=False)
        self.v = nn.Linear(feature_dim, hidden_size, bias=False)

    def forward(self, h, feats):
        if feats.dim()==2:
            feats = feats.unsqueeze(1)
        q = self.q(h)
        k = self.k(feats)
        v = self.v(feats)
        score = torch.einsum('bh,brh->br', q, k)
        attn = F.softmax(score, dim=1)
        ctx = torch.einsum('br,brh->bh', attn, v)
        return ctx

class YourDecoder(nn.Module):
    def __init__(self, vocab_size, feature_dim=512, hidden_size=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.attn = Attention(hidden_size, feature_dim)
        self.cell = nn.GRUCell(input_size=hidden_size*2, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def init_zero_hidden(self, batch, device):
        h0 = torch.zeros(batch, self.hidden_size, device=device)
        c0 = torch.zeros(batch, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, inputs, hidden_state, features):
        B, T = inputs.size()
        if features.dim()==3 and features.size(1)==1:
            features = features.squeeze(1)
        if hidden_state is None or hidden_state[0].size(0)!=B:
            h = torch.zeros(B, self.hidden_size, device=inputs.device)
        else:
            h = hidden_state[0]
        embs = self.embed(inputs)
        outs = []
        for t in range(T):
            ctx = self.attn(h, features)
            x = torch.cat([embs[:, t, :], ctx], dim=1)
            h = self.cell(x, h)
            outs.append(self.fc(h))
        logits = torch.stack(outs, dim=1)
        return logits, (h, torch.zeros_like(h))

    @torch.no_grad()
    def greedy_decode(self, features, max_len=50, start_id=1, end_id=2):
        if features.dim()==3 and features.size(1)==1:
            features = features.squeeze(1)
        B = features.size(0)
        device = features.device
        h = torch.zeros(B, self.hidden_size, device=device)
        cur = torch.full((B,), start_id, dtype=torch.long, device=device)
        tokens = []
        for _ in range(max_len):
            emb = self.embed(cur)
            ctx = self.attn(h, features)
            x = torch.cat([emb, ctx], dim=1)
            h = self.cell(x, h)
            logit = self.fc(h)
            cur = logit.argmax(dim=1)
            tokens.append(cur)
            if (cur==end_id).all():
                break
        if len(tokens)==0:
            return torch.empty(B, 0, dtype=torch.long, device=device)
        return torch.stack(tokens, dim=1)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1])
        vocab_size  = int(out_shape[0])
        hidden = 512
        self.encoder = YourEncoder(in_channels, hidden_dim=hidden)
        self.rnn     = YourDecoder(vocab_size, feature_dim=hidden, hidden_size=hidden)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = None
        self.vocab_size = vocab_size

    def _norm_caps(self, caps):
        if caps.ndim==3:
            caps = caps[:,0,:]
        elif caps.ndim==1:
            caps = caps.unsqueeze(0)
        return caps.long()

    def forward(self, images, captions=None, hidden_state=None):
        assert images.dim()==4
        feats = self.encoder(images)
        B = images.size(0)
        if captions is None:
            return self.rnn.greedy_decode(feats, max_len=50)
        caps = self._norm_caps(captions)
        inputs = caps[:, :-1]
        if hidden_state is None or (isinstance(hidden_state, tuple) and hidden_state[0].size(0)!=B):
            hidden_state = self.rnn.init_zero_hidden(B, images.device)
        logits, _ = self.rnn(inputs, hidden_state, feats)
        assert logits.dim()==3 and logits.size(1)==inputs.size(1)
        return logits

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            caps = self._norm_caps(captions)
            self.optimizer.zero_grad()
            logits = self(images, caps, None)
            T = min(logits.size(1), caps.size(1)-1)
            logits = logits[:, :T, :]
            tgt = caps[:, 1:1+T]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
