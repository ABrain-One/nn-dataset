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
        self.vocab_size = int(out_shape[0])  # Assuming output shape is (vocab_size,)
        self.hidden_dim = 640
        # TODO: Replace self.encoder with custom encoder producing memory tensor [B, S, H] where H >= 640
        
        # Example encoder implementation (CNN + MultiheadAttention)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Flatten the encoded features to get a sequence
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 640),
            nn.ReLU(),
            nn.Linear(640, 640)  # Final hidden dimension
        )
        
        # Resize the feature maps
        h, w = in_shape[2], in_shape[3]
        self.seq_length = ((h + 1) // 32) * ((w + 1) // 32)
        self.spatial_flattener = nn.PixelUnshuffle(2)  # Converts (C,H,W) to (CHW,H*W)
        
        # Calculate patch positions and normalize by channel means
        self.patch_positions = nn.Parameter(
            torch.rand(49, 512, requires_grad=True).float().to(device),
            requires_grad=True
        )
        
        self.encoder = nn.Sequential(
            self.backbone,
            self.global_pool,
            lambda x: self.spatial_flattener(x.squeeze(-1).transpose(0, 1)),
            self.feature_extractor
        )
        
        # TODO: Replace self.rnn with custom decoder implementing forward(inputs, hidden_state, features) -> (logits, hidden_state)
        self.transformer_dec = nn.TransformerDecoderLayer(
            d_model=640,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            norm_first=True
        )
        self.transformer_dec_net = nn.TransformerDecoder(
            self.transformer_dec,
            num_layers=3
        )
        
        self.embedding_layer = nn.Embedding(self.vocab_size, 640)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.embedding_dropout = nn.Dropout(0.1)
        self.final_proj = nn.Linear(640, self.vocab_size)
        
        self.max_sequence_length = 20
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)
        
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = self.criterion.to(self.device)
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            if captions.ndim == 3:
                caps = captions.long()
            else:
                caps = captions[:, 0, :].long()
                
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            images = images.to(self.device, dtype=torch.float32)
            memory = self.encoder(images)  # Shape: [B, S, 640]
            
            embedded_inputs = self.embedding_layer(inputs)
            embedded_inputs = self.embedding_dropout(embedded_inputs)
            logits = self.transformer_dec_net(embedded_inputs, memory, tgt_mask=None)
            logits = self.final_proj(logits)
            
            assert logits.shape == (images.size(0), inputs.size(1), self.vocab_size)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)  # Memory from encoder: [B, S, 640]
        if captions is not None:
            if captions.ndim == 3:
                captions = captions.long()
                
            tgt_input = captions[:, :-1]  # [B, T_in]
            tgt_output = captions[:, 1:]   # [B, T_out]
            
            embedded = self.embedding_layer(tgt_input)  # [B, T_in, 640]
            embedded = self.embedding_dropout(embedded)
            decoded = self.transformer_dec_net(tgt=embedded, memory=memory, memory_key_padding_mask=None, 
                                              tgt_mask=None)
            logits = self.final_proj(decoded)  # [B, T_in, vocab_size]
            
            return logits, hidden_state  # Shape: [B, T_in, vocab_size], None hidden_state
        else:
            raise ValueError("Generation mode requires specifying captions or enabling inference")