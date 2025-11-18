import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.embed = None
        self.sos_idx = kwargs.get('sos_idx', 1)
        self.eos_idx = kwargs.get('eos_idx', 2)
        self.vocab_size = out_shape[0]
        self.hidden_size = 768  # >=640

        # Encoder: Replace this with your custom encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.encoder.fc_proj = nn.Linear(512, self.hidden_size)

        # Decoder: Replace this with your custom decoder
        encoder_dim = self.hidden_size
        self.embed = nn.Embedding(self.vocab_size, encoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=encoder_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.proj = nn.Linear(encoder_dim, self.vocab_size)

    def train_setup(self, **kwargs):
        pass

    def learn(self, images, captions=None, **kwargs):
        pass

    def forward(self, images, captions=None, **kwargs):
        if captions is not None:
            tgt_input = captions[:, :-1]
            memory = self.encode(images)
            embedded = self.embed(tgt_input)
            embedded = embedded.transpose(0, 1)  # [T-1, B, 768]
            output = self.decoder(embedded, memory)
            output = output.transpose(0, 1)  # [B, T-1, 768]
            logits = self.proj(output)
            hidden_state = output[:,-1,:]  # [B, 768]
            return logits, hidden_state
        else:
            # Generate captions
            memory = self.encode(images)
            input_seq = torch.full((images.size(0), 1), self.sos_idx, device=images.device)
            embedded = self.embed(input_seq)
            max_length = 50  # Fixed maximum length for generation
            for _ in range(max_length-1):
                output = self.decoder(embedded, memory)
                logits = self.proj(output)
                next_token = logits.argmax(dim=-1)
                input_seq = torch.cat([input_seq, next_token], dim=1)
                embedded = self.embed(input_seq)
            return embedded, input_seq

    def encode(self, images):
        features = self.encoder(images)
        features = features.squeeze(2).squeeze(2)  # [B, 512]
        return self.encoder.fc_proj(features).unsqueeze(1)  # [B, 1, 768]

def supported_hyperparameters():
    return {'lr','momentum'}