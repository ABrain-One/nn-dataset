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
        self.vocab_size = int(out_shape)
        
        # Encoder with ResNet-like structure
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 768)
        )
        
        # Decoder: TransformerDecoder with cross-attention
        self.decoder_embedding = nn.Embedding(self.vocab_size, 768)
        self.decoder_pos_encoding = nn.Parameter(torch.zeros(50, 768))  # Learnable positional encoding
        
        # Transformer Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(encoder_layer, num_layers=6)
        
        # Final projection to vocabulary space
        self.projection = nn.Linear(768, self.vocab_size)
        
        # Set hidden dimension
        self.hidden_dim = 768
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        return None
    
    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions.to(self.device) if captions.ndim == 2 else captions[:, :, 0].to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            embedded_inputs = self.decoder_embedding(inputs)
            embedded_inputs = embedded_inputs + self.decoder_pos_encoding[:inputs.size(1), :][:None, :, None]
            
            out = self.transformer_decoder(embedded_inputs, memory)
            logits = self.projection(out)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            # Convert integer captions to embeddings
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            
            embedded_inputs = self.decoder_embedding(inputs)
            embedded_inputs = embedded_inputs + self.decoder_pos_encoding[:inputs.size(1), :][:None, :, None]
            
            out = self.transformer_decoder(embedded_inputs, memory)
            logits = self.projection(out)
            
            assert logits.shape == (images.size(0), inputs.shape[1], self.vocab_size)
            assert logits.shape[-1] == self.vocab_size
            
            return logits, hidden_state
        else:
            # Beam search implementation
            raise NotImplementedError("Beam search implementation required")