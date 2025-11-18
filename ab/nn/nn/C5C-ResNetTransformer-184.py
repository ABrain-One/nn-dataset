import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        
        # Encoder backbone replacement
        self.encoder = self.build_cnn_encoder()
        
        # Decoder parameters
        self.hidden_dim = 768
        self.num_layers = 2
        
        # Decoder replacement
        self.rnn = self.build_transformer_decoder()
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def build_cnn_encoder(self):
        """Build a CNN encoder that extracts features"""
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )
        return encoder
    
    def build_transformer_decoder(self):
        """Build a transformer decoder"""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        return nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.zeros((self.num_layers, batch, self.hidden_dim), device=device)
    
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions.to(self.device) if captions.ndim == 3 else captions[:, :, 0].to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            logits, _ = self.rnn(inputs, memory)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            cap_inputs = captions[:, 0, :].to(self.device)
            inputs = captions[:, :-1].to(self.device)
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state
        else:
            hidden_state = self.init_zero_hidden(images.size(0), self.device)
            return self.rnn(None, hidden_state, memory), hidden_state