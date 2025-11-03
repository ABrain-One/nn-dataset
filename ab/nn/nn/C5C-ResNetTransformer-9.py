import torch
import torch.nn as nn
import math

def supported_hyperparameters():
    return {'lr','momentum'}


class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 groups=1, norm=None, activation=None, dilation=1):
        super().__init__()
        if norm is not None:
            norm_layer = nn.BatchNorm2d(out_channels)
        else:
            norm_layer = None
        if activation is not None:
            if activation == "relu":
                act_layer = nn.ReLU
            elif activation == "silu":
                act_layer = nn.SiLU
            else:
                raise ValueError("Activation not supported")
        else:
            act_layer = None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, groups=groups, 
                              dilation=dilation, bias=False)
        self.norm = norm_layer
        self.act = act_layer

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.hidden_dim = 768

        image_size = in_shape[2]
        patch_size = get_closest_split(image_size, int(image_size * prm['patch_size']))

        self.stem = self.build_stem(in_shape[1], image_size, patch_size)

        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.pos_encoder_memory = PositionalEncoding(self.hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, 
                                                   dim_feedforward=3072, batch_first=True,
                                                   dropout=0.1, activation='relu', 
                                                   attention_dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)

        decoder_layers = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, 
                                                   dim_feedforward=3072, batch_first=True,
                                                   dropout=0.1, activation='relu', 
                                                   attention_dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=6)

        self.embedder = nn.Embedding(out_shape[0], self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, out_shape[0])

    def build_stem(self, in_channels, image_size, patch_size):
        stem = nn.Sequential()
        prev_channels = in_channels

        stem.add_module("stage1", Conv2dNormActivation(prev_channels, 64, 3, 1, 1))
        prev_channels = 64

        stem.add_module("stage2", Conv2dNormActivation(prev_channels, 128, 3, 2, 1))
        prev_channels = 128

        stem.add_module("stage3", Conv2dNormActivation(prev_channels, 256, 3, 1, 1))
        prev_channels = 256

        stem.add_module("stage4", Conv2dNormActivation(prev_channels, 512, 3, 1, 1))
        prev_channels = 512

        stem.add_module("stage5", Conv2dNormActivation(prev_channels, 768, 3, 1, 1))
        return stem

    def train_setup(self, prm):
        self.train()

    def learn(self, prm):
        return self.parameters()

    def forward(self, images, captions=None, **kwargs):
        images = images.to(self.device)
        B = images.shape[0]
        C = images.shape[1]
        H = images.shape[2]
        W = images.shape[3]

        memory = self.stem(images)
        n_h = memory.shape[-2]
        n_w = memory.shape[-1]
        S = n_h * n_w

        memory = memory.reshape(B, self.hidden_dim, S)
        memory = memory.permute(0, 2, 1)
        memory = self.pos_encoder_memory(memory)

        if captions is not None:
            captions = captions.to(self.device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            embedded = self.embedder(inputs)
            embedded = embedded.permute(0, 1, 2)
            embedded = self.pos_encoder(embedded)

            output = self.transformer_decoder(tgt=embedded, memory=memory)
            output = output.permute(0, 1, 2)
            logits = self.proj(output)
            hidden_state = output

            return logits, hidden_state
        else:
            return None

def get_closest_split(size, factor):
    desired = size // factor
    # Find the closest divisor to desired
    divisors = []
    for i in range(1, int(math.sqrt(desired)) + 1):
        if desired % i == 0:
            divisors.append(i)
            divisors.append(desired // i)
    divisors.sort()
    closest = min(divisors, key=lambda x: abs(x - desired))
    return closest