import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr','momentum'}



class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):

            # ---- API aliases (auto-injected) ----
            self.in_shape = in_shape
            self.out_shape = out_shape
            self.device = device
            self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
            self.vocab_size = out_shape[0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
            self.out_dim = self.vocab_size
            self.num_classes = self.vocab_size

            # Backward-compat local aliases (old LLM patterns)
            vocab_size = self.vocab_size
            out_dim = self.vocab_size
            num_classes = self.vocab_size
            in_channels = self.in_channels
super(Net, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        # Encoder: Vision Transformer style
        self.encoder = VisionTransformerEncoder(in_channels=in_shape[1], hidden_size=768)
        
        # Decoder: LSTM-based
        self.decoder = LSTMDecoder(hidden_size=768, output_size=out_shape[0])

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    def learn(self, train_data):
        # This function is called for each training step.
        # It should update the model's parameters based on the training data.
        pass

    def forward(self, images, captions=None, hidden_state=None):
        # This function should return the output of the model.
        # If captions are provided, it should use teacher forcing.
        pass


class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=768, num_layers=4, num_heads=8):
        super(VisionTransformerEncoder, self).__init__()
        self.num_patches = 14*14  # 196
        self.patch_size = 16
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Project the image into patches
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=16, stride=16)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
            self._layer_list.append(layer)

        # Final norm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Project the image into patches
        x = self.proj(x)  # [B, hidden_size, H//16, W//16]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        return x


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.embed = nn.Embedding(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden_state=None, features=None):
        # inputs: [B, S] (sequence of word indices)
        # hidden_state: [B, hidden_size] (initial hidden state)
        # features: [B, num_patches, hidden_size] (the encoder features)

        # If hidden_state is None, we initialize it.
        if hidden_state is None:
            hidden_state = self.init_hidden(inputs.size(0))

        # Embed the inputs
        embedded = self.embed(inputs)  # [B, S, hidden_size]

        # If features are provided, we use them for attention.
        # But the decoder's forward should return the output and the hidden_state.
        # For simplicity, we'll just use the embedded inputs.

        outputs, hidden_state = self.lstm(embedded, hidden_state)
        logits = self.fc(outputs)

        return logits, hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)