import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_act(self.expand1x1(x)), self.expand3x3_act(self.expand3x3(x))], 1
        )

class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.stage1 = Fire(64, 16, 64, 64)
        self.stage2 = Fire(128, 16, 64, 64)
        self.stage3 = Fire(128, 32, 128, 128)
        self.stage4 = Fire(256, 32, 128, 128)
        self.stage5 = Fire(384, 48, 192, 192)
        self.stage6 = Fire(384, 64, 256, 256)
        self.stage7 = Fire(512, 64, 256, 256)
        
        self.final_bn = nn.BatchNorm2d(hidden_dim)
        self.final_dropout = nn.Dropout(0.5)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(512, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial processing
        x = F.relu(self.bn0(self.stem(x)))
        x = F.max_pool2d(x, 3, 2, 1)
        
        # Sequential processing through stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        # Final processing
        x = self.final_pool(x)
        x = self.final_bn(x)
        x = F.relu(x)
        x = self.final_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, vocab_size: int = None) -> None:
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, hidden_dim))
        self.transformer_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads)
        self.transformer = nn.TransformerDecoder(self.transformer_layers, num_layers=6, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed and add positional encoding
        embedded = self.embedder(inputs)
        pos = self.pos_encoder.repeat(embedded.size(0), 1, 1)
        embedded = embedded + pos
        
        # Process through transformer
        memory_key_padding_mask = None  # All memory positions active
        out = self.transformer(
            embedded,
            memory,
            None,  # key_padding_mask for memory (optional)
            None   # mask for tgt (optional)
        )
        
        # Final prediction
        logits = self.fc_out(out)
        return logits, None

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 768
        
        # Get input properties
        channel_number = in_shape[1]
        height = in_shape[2]
        width = in_shape[3]
        
        # Initialize encoder and decoder
        self.encoder = Encoder(channel_number, hidden_dim=self.hidden_dim)
        self.decoder = Decoder(hidden_dim=self.hidden_dim, num_heads=min(8, self.hidden_dim//64), vocab_size=self.vocab_size)

    def init_zero_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.tensor([]).to(device), torch.tensor([]).to(device)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm.get('lr', 1e-3)))
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def learn(self, train_data) -> None:
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            caps = captions[:,0,:] if captions.ndim == 3 else captions
            inputs = caps[:, :-1]
            targets = caps[:, 1:] if captions.ndim == 3 else caps[:, 1]
            
            self.optimizer.zero_grad()
            memory = self.encoder(images)
            logits, _ = self.decoder(inputs, memory)
            
            assert images.dim() == 4
            assert logits.shape == (inputs.shape[0], inputs.shape[1], self.vocab_size)
            assert targets.shape == (targets.size(0), targets.size(1))
            
            loss = self.criterion(logits.permute(0, 2, 1).contiguous().view(-1, self.vocab_size), 
                                 targets.reshape(-1).to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, 
                hidden_state: Optional[torch.Tensor] = None) -> tuple:
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            if captions.ndim == 3:
                targets = captions[:, 1:].reshape(-1)
                logits, _ = self.decoder(captions[:, :-1], memory, hidden_state)
                return logits, None
                
        raise NotImplementedError("Generation without captions not implemented yet.")

    def generate_caption(self, images, ...) -> ...:
        # Beam search or simple generation
        # Implementation pending
        pass

def supported_hyperparameters():
    return {'lr','momentum'}