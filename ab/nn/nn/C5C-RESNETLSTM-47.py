import math
from typing import Callable, Tuple, Optional
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def supported_hyperparameters():
    return {'lr', 'momentum'}


class ConvStemConfig:
    def __init__(self, out_channels, kernel_size, stride, norm_layer, activation_layer):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = lambda d: nn.LayerNorm(d, eps=1e-6),
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.ln1 = norm_layer(hidden_dim)
        self.ln2 = norm_layer(hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.ff_block = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Skip connection
        residue = x

        # Pre-normalization
        x = self.ln1(x)

        # Multi-head self-attention
        x, _ = self.attention(x, x, x)

        # Post-attention dropout (0 or dropout depending on train/eval)
        x = F.dropout(x, p=(0.3 if self.training else 0.0))

        # Add skip connection (only during training to mirror original intent)
        x = residue + x if self.training else x

        # Pre-normalization
        x = self.ln2(x)

        # Fully connected layer
        x = self.ff_block(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Stacked layers
        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # Add positional embedding
        x = x + self.pos_embedding

        # Apply dropout
        x = self.dropout(x)

        # Pass through all blocks
        x = self.blocks(x)

        return x


def get_divisors(n, res=None):
    res = res or []
    i = 1
    while i <= n:
        if n % i == 0:
            res.append(i)
        i += 1
    return res


def get_closest_split(n, close_to):
    divisors = get_divisors(n)
    closest = None

    for divisor in divisors:
        diff = abs(divisor - close_to)
        if closest is None or diff < abs(closest - close_to):
            closest = divisor

    return closest


class Net(nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        out_shape: tuple,
        prm: dict,
        device: torch.device,
        hidden_dim_base: int = 768,
        num_heads_base: int = 8,
    ) -> None:
        super().__init__()

        # Store hyperparameters
        self.device = device
        self.hidden_dim_base = max(int(hidden_dim_base), 640)
        self.num_heads_base = num_heads_base

        # Input/output shape parameters
        channel_number = int(in_shape[1]) if len(in_shape) >= 2 else 3

        # Robust vocab_size extraction (handles nested tuples/lists or plain int)
        def _first_int(x):
            if isinstance(x, (tuple, list)):
                return _first_int(x[0])
            return int(x)

        vocab_size = _first_int(out_shape)

        image_size = int(in_shape[2]) if len(in_shape) >= 3 else 224

        # Extract hyperparameters safely
        self.dropout = float(min(max(prm.get('dropout', 0.2), 0.1), 0.3))
        self.attention_dropout = float(min(max(prm.get('attention_dropout', 0.1), 0.05), 0.2))
        self.patch_size = int(get_closest_split(image_size, image_size * self.dropout))
        self.lr_factor = 0.8

        # Derived sizes
        self.hidden_dim = self.hidden_dim_base
        self.decoder_hidden_dim = self.hidden_dim_base

        # CBAM-like attention trunk (fixed activations instantiation and channels)
        self.channel_reduction = 32
        self.cbam = nn.Sequential(
            nn.Conv2d(self.channel_reduction * 2, self.hidden_dim_base, 1),
            nn.BatchNorm2d(self.hidden_dim_base),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_base, self.hidden_dim_base, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim_base),
            nn.ReLU(inplace=True),
        )

        # Encoder CNN (ensure ReLU modules are instantiated)
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_number, self.channel_reduction, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.channel_reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_reduction, self.channel_reduction * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_reduction * 2),
            nn.ReLU(inplace=True),
            self.cbam,
        )

        # Feature pooling + projection (fix in_features to match cbam output channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_fc = nn.Linear(self.hidden_dim_base, self.hidden_dim)

        # Decoder - LSTM
        self.embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.num_decoder_layers = max(int(12 * (1 - self.dropout)), 3)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_decoder_layers,
            dropout=min(self.dropout, 0.3) if self.num_decoder_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        # Set learning rate and momentum
        self.learning_rate = float(prm.get('lr', 1e-3))
        self.momentum = float(prm.get('momentum', 0.9))

        # Training helpers
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999)
        )
        self.iteration_count = 0

        self.to(self.device)

    def _init_lstm_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_decoder_layers, batch_size, self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.num_decoder_layers, batch_size, self.hidden_dim, device=self.device)
        return h0, c0

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        B = images.size(0)

        # Image processing branch (encoder)
        features = self.cnn(images)  # [B, hidden_dim_base, H', W']
        features = self.global_pool(features)  # [B, hidden_dim_base, 1, 1]
        features = torch.flatten(features, 1)  # [B, hidden_dim_base]
        features = F.relu(self.feature_fc(features))  # [B, hidden_dim]

        if hidden_state is None:
            hidden_state = self._init_lstm_hidden(B)

        if captions is not None:
            # Teacher forcing: shift inputs/targets
            inputs = captions[:, :-1]
            # Embed inputs and combine with image features (cross conditioning)
            embedded = self.embedding(inputs)  # [B, T-1, H]
            embedded = embedded + features.unsqueeze(1)  # broadcast to time steps
            outputs, hidden_state = self.lstm(embedded, hidden_state)
            logits = self.fc(F.dropout(outputs, p=(self.dropout if self.training else 0.0)))
            return logits, hidden_state
        else:
            # Greedy generation (kept simple)
            max_len = 20
            sos_idx = 1
            input_seq = torch.full((B, 1), sos_idx, device=self.device, dtype=torch.long)
            embedded = self.embedding(input_seq) + features.unsqueeze(1)
            outputs_all = []

            for _ in range(max_len - 1):
                out_step, hidden_state = self.lstm(embedded, hidden_state)
                logits = self.fc(out_step)
                preds = logits.argmax(dim=-1)  # [B, 1]
                outputs_all.append(logits)
                embedded = self.embedding(preds) + features.unsqueeze(1)

            # Concatenate logits over time: [B, T-1, V]
            if outputs_all:
                logits_seq = torch.cat(outputs_all, dim=1)
            else:
                logits_seq = torch.empty(B, 0, self.fc.out_features, device=self.device)
            return logits_seq, hidden_state

    def train_setup(self, prm):
        # Keep optimizer/criterion configurable
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(prm.get('lr', self.learning_rate)),
            betas=(float(prm.get('momentum', self.momentum)), 0.999),
        )
        self.to(self.device)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)

            hidden = self._init_lstm_hidden(batch_size)
            logits, _ = self.forward(inputs, labels, hidden)

            # Reshape for token-level cross-entropy
            logits_reshaped = logits.reshape(-1, logits.size(-1))
            labels_reshaped = labels[:, 1:].reshape(-1)  # align with logits (shifted)

            loss = self.criteria[0](logits_reshaped, labels_reshaped)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            if (self.iteration_count % 5) == 0:
                print(f"Iteration {self.iteration_count}: Loss = {loss.item():.4f}")
            self.iteration_count += 1


if __name__ == "__main__":
    # Simple compile-time check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_shape = (3, 3, 224, 224)
    out_shape = (5000,)
    net = Net(in_shape, out_shape, {'lr': 0.001, 'momentum': 0.9}, device)

    test_images = torch.rand(4, 3, 224, 224, device=device)
    test_captions = torch.randint(0, 5000, (4, 20), device=device)

    net.train_setup({'lr': 0.001, 'momentum': 0.9})
    net.learn([(test_images, test_captions)])
