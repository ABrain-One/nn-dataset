import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler



def supported_hyperparameters():
    # Supported hyperparameters for Optuna or manual tuning
    return {'lr', 'momentum'}

# --- Spatial Attention Encoder ---
class ResNetSpatialEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(backbone.children())[:-2]  # Keep conv layers, remove pool & fc
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, output_dim)
        # Optionally freeze backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)             # [B, 2048, 7, 7]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W)       # [B, 2048, 49]
        x = x.permute(0, 2, 1)      # [B, 49, 2048]
        x = self.fc(x)              # [B, 49, output_dim]
        return x                    # [B, num_regions, output_dim]

# --- Spatial Attention LSTM Decoder ---
class SpatialAttentionLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, feature_dim=768, hidden_size=768, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn_linear = nn.Linear(feature_dim + hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.lstm = nn.LSTMCell(hidden_size + feature_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, features, captions, hidden_state=None):

        B, num_regions, feature_dim = features.size()
        seq_len = captions.size(1)

        # Init hidden/cell state if not provided
        if hidden_state is None:
            h = features.mean(dim=1)  # [B, feature_dim]
            h = torch.tanh(h)
            c = torch.zeros(B, self.hidden_size, device=features.device)
        else:
            h, c = hidden_state

        embeddings = self.embedding(captions)  # [B, seq_len, hidden_size]
        outputs = []
        for t in range(seq_len):
            emb_t = embeddings[:, t, :]  # [B, hidden_size]

            # Attention
            h_exp = h.unsqueeze(1).expand(-1, num_regions, -1)   # [B, num_regions, hidden]
            attn_input = torch.cat([features, h_exp], dim=2)     # [B, num_regions, feat+hidden]
            attn_hidden = torch.tanh(self.attn_linear(attn_input))  # [B, num_regions, hidden]
            attn_scores = self.attn_v(attn_hidden).squeeze(2)    # [B, num_regions]
            alpha = torch.softmax(attn_scores, dim=1)            # [B, num_regions]
            context = (features * alpha.unsqueeze(2)).sum(dim=1) # [B, feature_dim]

            # LSTM step
            lstm_input = torch.cat([emb_t, context], dim=1)      # [B, hidden+feature]
            h, c = self.lstm(lstm_input, (h, c))
            out_t = self.fc(self.dropout(h))                     # [B, vocab_size]
            outputs.append(out_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [B, seq_len, vocab_size]
        return outputs, (h, c)

    # For step-by-step (greedy) inference
    def step(self, input_token, features, h, c):
        # input_token: [B]
        emb = self.embedding(input_token)      # [B, hidden_size]

        # Attention
        B, num_regions, feature_dim = features.size()
        h_exp = h.unsqueeze(1).expand(-1, num_regions, -1)
        attn_input = torch.cat([features, h_exp], dim=2)
        attn_hidden = torch.tanh(self.attn_linear(attn_input))
        attn_scores = self.attn_v(attn_hidden).squeeze(2)
        alpha = torch.softmax(attn_scores, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)

        # LSTM step
        lstm_input = torch.cat([emb, context], dim=1)
        h, c = self.lstm(lstm_input, (h, c))
        logits = self.fc(self.dropout(h))
        return logits, (h, c)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode='max'):

        self.patience = patience            # patience: epochs to wait after last improvement
        self.min_delta = min_delta          # min_delta: minimum BLEU improvement to count as better
        self.mode = mode                    # mode: 'max' for BLEU (higher is better)
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

# --- Main Model Class ---
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = 768
        self.vocab_size = out_shape[0]
        self.cnn = ResNetSpatialEncoder(self.hidden_size)
        self.rnn = SpatialAttentionLSTMDecoder(
            self.vocab_size,
            feature_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,         # One LSTM layer for spatial attention is standard
            dropout=0.3
        )

    def forward(self, images, captions=None, hidden_state=None):
        features = self.cnn(images)  # [B, num_regions, feat_dim]
        batch_size = features.size(0)

        # Init hidden state
        h = features.mean(dim=1)    # [B, hidden_size]
        h = torch.tanh(h)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        hidden_state = (h, c)

        if captions is not None:
            # Teacher forcing (training)
            sos_idx = {v: k for k, v in self.__class__.idx2word.items()}.get('<SOS>', 1) if hasattr(self.__class__, 'idx2word') else 1
            sos_token = torch.full((captions.size(0), 1), sos_idx, dtype=torch.long, device=self.device)
            inputs = torch.cat([sos_token, captions[:, :-1]], dim=1)
            targets = captions
            outputs, _ = self.rnn(features, inputs, hidden_state)
            return outputs, targets
        else:
            # Greedy decoding for inference
            max_len = 20
            sos_idx = {v: k for k, v in self.__class__.idx2word.items()}.get('<SOS>', 1) if hasattr(self.__class__, 'idx2word') else 1
            inputs = torch.full((batch_size,), sos_idx, dtype=torch.long, device=self.device)
            h, c = hidden_state
            captions_out = []
            for _ in range(max_len):
                logits, (h, c) = self.rnn.step(inputs, features, h, c)
                predicted = logits.argmax(1)
                captions_out.append(predicted.unsqueeze(1))
                inputs = predicted
            return torch.cat(captions_out, dim=1)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'])
        self.scaler = GradScaler()

    def learn(self, train_data):
        self.train()
        for i, (images, captions) in enumerate(train_data):
            images = images.to(self.device)
            captions = captions.to(self.device)
            B, C, H, W = images.shape
            N_CAP = captions.shape[1]
            images_exp = images.repeat_interleave(N_CAP, dim=0)
            captions_exp = captions.reshape(-1, captions.shape[-1])

            self.optimizer.zero_grad()
            # --- AMP autocast context ---
            with autocast(device_type=self.device.type):
                outputs, targets = self.forward(images_exp, captions_exp)
                loss = self.criteria[0](
                    outputs.contiguous().view(-1, outputs.shape[2]),
                    targets.contiguous().view(-1)
                )
            # --- AMP scaler logic ---
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Print loss for every 300 batches (optional)
            if i % 300 == 0:
                print(f"Batch {i}: Loss: {loss.item():.4f}")


    def eval_mode_generate_captions(self, images):
        return self.forward(images)
