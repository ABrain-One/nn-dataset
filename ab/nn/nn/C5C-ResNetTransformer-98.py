import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -2 * math.log(1000) / d_model)
        if d_model % 2 == 1:
            # zero-padding div_term
            pass
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, :] = torch.sin(position.double() * div_term)
        pe[:, 0, 1::2] = torch.cos(position.double() * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ImageEncoder(nn.Module):
    def __init__(self, in_shape, out_features=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.transformer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, hidden_state, memory):
        embedded = self.embedding(input_ids)
        embedded = embedded + self.pos_encoder(embedded)
        mask = torch.triu(torch.ones(embedded.size(1), embedded.size(1)), diagonal=1)
        mask = mask.masked_fill(mask == 0, -10000.0)
        out = self.decoder(embedded, memory, tgt_mask=mask)
        logits = self.fc(out)
        if hidden_state is None:
            hidden_state = memory.squeeze(1)
        else:
            hidden_state = memory.squeeze(1)
        return logits, hidden_state

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.encoder = ImageEncoder(in_shape, out_features=prm['hidden_dim'])
        self.decoder = Decoder(vocab_size=self.vocab_size, d_model=prm['hidden_dim'], num_layers=prm['num_layers'], nhead=prm['nhead'])
        self.embedding = nn.Embedding(self.vocab_size, prm['hidden_dim'])

    def train_setup(self, optimizer, lr, momentum):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            if 'momentum' in param_group:
                param_group['momentum'] = momentum

    def learn(self, train_data):
        images = train_data['images']
        captions = train_data['captions']
        images = images.to(self.device, dtype=torch.float32)
        captions = captions.to(self.device)
        memory = self.encoder(images)
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        logits, hidden_state = self.decoder(inputs, None, memory)
        loss = self.calculate_loss(logits, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def calculate_loss(self, logits, targets):
        batch_size = logits.size(0)
        seq_length = logits.size(1)
        logits = logits.view(batch_size * seq_length, -1)
        targets = targets.view(batch_size * seq_length)
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, targets)

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        if captions is not None:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            logits, hidden_state = self.decoder(inputs, hidden_state, memory)
            return logits, hidden_state
        else:
            if hidden_state is None:
                hidden_state = memory.squeeze(1)
            return memory, hidden_state

def supported_hyperparameters():
    return {'lr', 'momentum'}