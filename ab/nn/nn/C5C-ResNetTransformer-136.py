import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 640  # Must be >=640
        
        # Encoder section: Replace with custom CNN backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(int(in_shape[1]), 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(start_dim=1)  # This converts image features to tokens
        )
        
        # Decoder section: Using nn.TransformerDecoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=min(8, self.hidden_dim//64),
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Post-decoder processing layer
        self.post_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.vocab_size)
        )
        
        # Initialize encoder with proper weight scaling
        self._initialize_encoder_weights()
        
        # Teacher forcing components
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(50, 1, self.hidden_dim), requires_grad=True)
        self.optimizer = None
        
    def _initialize_encoder_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.01)
                    
        # Weight initialization for post_decoder needs careful handling
        nn.init.xavier_uniform_(self.post_decoder[0].weight)
        nn.init.xavier_uniform_(self.post_decoder[2].weight)
        if self.post_decoder[0].bias is not None:
            nn.init.constant_(self.post_decoder[0].bias, 0)
        if self.post_decoder[2].bias is not None:
            nn.init.normal_(self.post_decoder[2].bias, 0.01)
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        """Initialize decoder hidden state"""
        return torch.zeros(1, batch, self.hidden_dim).to(device)
        
    def train_setup(self, prm: dict):
        """Set up training configuration"""
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('beta1', 0.9)), 0.9), 0.99)
        if 'momentum' in prm:
            momentum = float(prm['momentum'])
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.criteria = [self.criterion]
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
                out = self.forward(images, captions)
                logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def forward(self, images, captions):
        # Encoder
        encoder_output = self.encoder(images)
        # Project to hidden_dim
        encoder_output = encoder_output.reshape(encoder_output.size(0), -1, self.hidden_dim)
        
        # Prepare the decoder input: captions
        # We need to embed the captions and add the positional encoding.
        # First, let's get the caption indices and their shape.
        if captions.ndim == 3:
            caption_indices = captions[:,0,:]   # This is the first time step for the entire batch
            caption_seq = captions.size(1) - 1    # Because we will remove the first time step for the target
        else:
            caption_indices = captions[:,0]        # This is the first time step for the entire batch
            caption_seq = captions.size(1) - 1
        
        # Now, embed the caption_indices
        embedded_captions = self.embedding(caption_indices)   # (batch, caption_seq, hidden_dim)
        
        # Add the positional encoding
        # The pos_encoder is of shape (50, 1, hidden_dim). We will use the first `caption_seq` steps.
        pos_encoder = self.pos_encoder[:caption_seq, :]
        embedded_captions = embedded_captions + pos_encoder
        
        # Now, pass to the decoder
        decoder_output = self.decoder(embedded_captions, encoder_output)
        
        # Then, pass through the post_decoder
        output = self.post_decoder(decoder_output)
        
        return output