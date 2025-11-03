import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 640
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, self.hidden_dim)
        )
        
        # Decoder
        self.embed = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8),
            num_layers=6,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
    def init_zero_hidden(self, batch, device):
        return torch.empty(0, device=device), torch.empty(0, device=device)
        
    def learn(self, images, captions):
        assert images.dim() == 4
        assert captions.dim() == 3
        assert images.shape[0] == captions.shape[0]
        assert images.shape[2] == captions.shape[1]
        assert self.vocab_size == captions.shape[2]
        
        # We assume captions are integer indices.
        # We'll use teacher forcing.
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        memory = self.encoder(images)
        embedded = self.embed(inputs)
        out = self.transformer_decoder(embedded, memory)
        logits = self.fc_out(out)
        hidden_state = None
        return logits, hidden_state
        
    def forward(self, images, captions=None):
        if captions is not None:
            # Use teacher forcing
            return self.learn(images, captions)
        else:
            # Generate captions
            pass

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

def supported_hyperparameters():
    return {'lr','momentum'}