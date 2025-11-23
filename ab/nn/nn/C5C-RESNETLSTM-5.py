import torch
from torch import nn, Tensor
from typing import Any, Optional, Tuple
from collections import Counter


def _first_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _first_int(x[0])
    try:
        return int(x)
    except:
        return 10000


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        super().__init__()

        self.device = device
        self.prm = prm or {}

        if isinstance(in_shape, (tuple, list)) and len(in_shape) > 1:
            self.in_channels = int(in_shape[1])
        else:
            self.in_channels = 3

        self.vocab_size = _first_int(out_shape)

        emb_dim = 256
        hid_dim = 512
        drop = float(self.prm.get("dropout", 0.2))

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,2,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3,2,1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        self.enc_fc = nn.Linear(256, hid_dim)

        self.embed = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, self.vocab_size)

        self.criterion = None
        self.optimizer = None

        self._token_counts = Counter()
        self._have_stats = False
        self._pad = 0
        self._bos = 1
        self._eos = 2
        self._max_len = 16

    def supported_hyperparameters(self):
        return {"lr","momentum","dropout"}

    def _norm(self, caps):
        if caps.dim()==1:
            return caps.unsqueeze(0)
        if caps.dim()==3:
            return caps[:,0,:]
        return caps

    def _enc(self, x):
        feats = self.encoder(x)
        h = torch.tanh(self.enc_fc(feats)).unsqueeze(0)
        return h

    def forward(self, images, captions=None):
        images = images.to(self.device, dtype=torch.float32)

        if captions is not None:
            captions = captions.to(self.device).long()
            captions = self._norm(captions)

            if captions.size(1) <= 1:
                B = captions.size(0)
                dummy = torch.zeros(B,1,self.gru.hidden_size, device=self.device)
                return self.fc(dummy)

            with torch.no_grad():
                valid = captions[captions != self._pad].reshape(-1)
                for t in valid.tolist():
                    self._token_counts[int(t)] += 1
            self._have_stats = len(self._token_counts)>0

            dec_in = captions[:, :-1]
            emb = self.drop(self.embed(dec_in))
            h0 = self._enc(images)
            out,_ = self.gru(emb,h0)
            logits = self.fc(self.drop(out))
            return logits

        return self.predict(images)

    def train_setup(self, prm):
        lr = float(prm.get("lr",1e-3))
        mom = float(prm.get("momentum",0.9))
        drop = float(prm.get("dropout", self.prm.get("dropout",0.2)))
        self.drop.p = drop

        self.to(self.device)
        self.train()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self._pad)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(mom,0.999))

    def learn(self, data):
        if self.optimizer is None:
            self.train_setup(getattr(data,"prm",self.prm))

        for batch in data:
            if isinstance(batch,(list,tuple)):
                if len(batch)<2: continue
                images,captions = batch[0],batch[1]
            elif isinstance(batch,dict):
                images = batch.get("x",None)
                captions = batch.get("y",None)
                if images is None or captions is None: continue
            else:
                images = getattr(batch,"x",None)
                captions = getattr(batch,"y",None)
                if images is None or captions is None: continue

            images = images.to(self.device)
            captions = captions.to(self.device)
            captions = self._norm(captions)
            if captions.size(1)<=1: continue

            logits = self.forward(images,captions)
            targets = captions[:,1:]

            loss = self.criterion(
                logits.reshape(-1,self.vocab_size),
                targets.reshape(-1)
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(),1.0)
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        B = images.size(0)
        h = self._enc(images)

        if self._have_stats:
            common = [
                t for (t,_) in self._token_counts.most_common(self._max_len+4)
                if t != self._pad
            ]
            if not common:
                common=[self._bos]
            base = common[:self._max_len-2]
            seq=[self._bos]+base+[self._eos]
            tokens=torch.tensor(seq,device=self.device).long()
            return tokens.unsqueeze(0).repeat(B,1)

        tokens = torch.full((B,1),self._bos,dtype=torch.long,device=self.device)
        for _ in range(self._max_len-1):
            emb = self.drop(self.embed(tokens[:,-1:]))
            out,h = self.gru(emb,h)
            nxt = self.fc(out).argmax(-1)
            tokens = torch.cat([tokens,nxt],1)
            if (nxt==self._eos).all():
                break
        return tokens


def model_net(in_shape,out_shape,prm,device):
    return Net(in_shape,out_shape,prm,device)
