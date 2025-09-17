import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Encoder(nn.Module):
    def __init__(self, in_channels: int, embed_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adap_pool = nn.AdaptiveMaxPool2d(7)
        self.fc = nn.Linear(128 * 7 * 7, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adap_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, drop_prob: float = 0.0, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size * 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)

    def init_zero_hidden(self, batch_size: int, device: torch.device):
        return self.init_hidden(batch_size, device)

    def forward(self, inputs: torch.Tensor, hidden: tuple, encoder_features: torch.Tensor):
        B, T = inputs.size()
        embeds = self.embed(inputs)
        h, c = hidden
        outputs = []
        for t in range(T):
            word_t = embeds[:, t, :]
            step_in = torch.cat([word_t, encoder_features], dim=1)
            out, (h, c) = self.lstm(step_in.unsqueeze(1), (h, c))
            outputs.append(self.fc(out.squeeze(1)))
        logits = torch.stack(outputs, dim=1)
        return logits, (h, c)

    @torch.no_grad()
    def generate(self, encoder_features: torch.Tensor, hidden: tuple, max_len: int = 50, start_token: int = 1, end_token: int = 2):
        device = encoder_features.device
        B = encoder_features.size(0)
        cur = torch.full((B,), start_token, dtype=torch.long, device=device)
        generated = []
        for _ in range(max_len):
            step_logits, hidden = self.forward(cur.unsqueeze(1), hidden, encoder_features)
            next_ids = step_logits.squeeze(1).argmax(dim=1)
            generated.append(next_ids)
            cur = next_ids
            if (next_ids == end_token).all():
                break
        if len(generated) == 0:
            return torch.empty(B, 0, dtype=torch.long, device=device), hidden
        return torch.stack(generated, dim=1), hidden

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = int(in_shape[1])
        self.vocab_size = int(out_shape[0])
        embed_size = 256
        hidden_size = 512
        self.encoder = Encoder(in_channels, embed_size)
        self.rnn = DecoderRNN(embed_size, hidden_size, self.vocab_size, drop_prob=0.0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = None

    @staticmethod
    def _normalize_captions(captions: torch.Tensor) -> torch.Tensor:
        if captions is None:
            return None
        if captions.ndim == 3:
            captions = captions[:, 0, :]
        elif captions.ndim == 1:
            captions = captions.unsqueeze(0)
        if captions.dtype != torch.long:
            captions = captions.long()
        return captions

    def _ensure_hidden(self, hidden_state, batch_size: int):
        if hidden_state is None:
            return self.rnn.init_zero_hidden(batch_size, self.device)
        h = hidden_state[0]
        if h.size(1) != batch_size:
            return self.rnn.init_zero_hidden(batch_size, self.device)
        return hidden_state

    def forward(self, images, captions=None, hidden_state=None):
        B = images.size(0)
        features = self.encoder(images)
        captions = self._normalize_captions(captions)
        hidden_state = self._ensure_hidden(hidden_state, B)
        if captions is not None:
            inputs = captions[:, :-1]
            logits, _ = self.rnn(inputs, hidden_state, features)
            return logits
        tokens, _ = self.rnn.generate(features, hidden_state, max_len=50)
        return tokens

    def train_setup(self, prm):
        self.to(self.device)
        lr = float(prm.get('lr', 1e-3)) if isinstance(prm, dict) else 1e-3
        momentum = float(prm.get('momentum', 0.9)) if isinstance(prm, dict) else 0.9
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions.to(self.device)
            caps_for_loss = self._normalize_captions(captions)
            self.optimizer.zero_grad()
            logits = self(images, captions, None)
            T_eff = caps_for_loss.size(1) - 1
            if logits.size(1) != T_eff:
                T_match = min(logits.size(1), T_eff)
                logits = logits[:, :T_match, :]
                caps_for_loss = caps_for_loss[:, :T_match + 1]
            loss = self.criterion(
                logits.contiguous().view(-1, self.vocab_size),
                caps_for_loss[:, 1:].contiguous().view(-1).long()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
