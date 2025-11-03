import torch.nn as nn
import torch.optim as optim
import torch
from typing import Optional, Tuple

class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: Tuple[int, int, int], prm: dict, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm  # Store prm as an instance variable
        
        # Encoder: produce memory features [B, S, H] with H>=640
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * (in_shape[2]//4) * (in_shape[3]//4), 768)
        )
        
        # Decoder: use transformer decoder
        self.decoder = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder, num_layers=3)
        
        # Final output layer
        self.fc_out = nn.Linear(768, out_shape[2])
        
        # For teacher forcing
        self.vocab_size = out_shape[2]
        self.embeddings = nn.Embedding(self.vocab_size, 768)
        
    def train_setup(self, opt: optim.Optimizer) -> None:
        opt.add_param_group({'params': self.parameters(), 'lr': self.prm['lr']})
        
    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert images.dim() == 4
        assert captions.dim() == 2 or captions.dim() == 3
        
        if captions.dim() == 3:
            caps = captions[:, 0, :].long().to(self.device)
        else:
            caps = captions.long().to(self.device)
            
        inputs = caps[:, :-1]
        targets = caps[:, 1:]
        
        memory = self.encoder(images)
        embedded = self.embeddings(inputs) * (self.embeddings.embedding_dim ** 0.5)
        embedded = embedded.permute(1, 0, 2)
        
        memory = memory.permute(1, 0, 2)
        
        out = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            memory_mask=None,
            tgt_mask=None
        )
        
        logits = self.fc_out(out.transpose(0, 1))
        return logits
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if captions is None:
            # Inference mode
            memory = self.encoder(images)
            return None, memory
        
        # Teacher forcing mode
        return self.learn(images, captions)

def supported_hyperparameters():
    return {'lr','momentum'}



# --- auto-closed by AlterCaptionNN ---
)