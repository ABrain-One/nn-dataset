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
        
        # Extract vocab size from out_shape
        self.vocab_size = int(out_shape)
        
        # Initialize encoder
        self.encoder = PatchCNNEncoder(in_channels=in_shape[1], out_channels=768)
        
        # Initialize decoder
        self.decoder = TransformerDecoder(vocab_size=self.vocab_size, d_model=768, num_layers=6, num_heads=8)
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)
        
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            logits, _ = self.decoder(inputs, None, memory)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

class PatchCNNEncoder(nn.Module):
    """Simple CNN encoder that extracts patches and projects to 768 dimensions."""
    def __init__(self, in_channels=3, out_channels=768, device=torch.device('cpu')):
        super().__init__()
        
        # Stem layer
        self.stem = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
            activation_layer=nn.ReLU,
            inplace=True
        )
        
        # Intermediate layers
        self.layer1 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.layer2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # Final projection for patches
        self.patch_proj = Conv2dNormActivation(out_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.patch_proj(x)
        
        # Flatten spatial dimensions while keeping channels intact
        batch_size, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        
        # Transpose to get [B*S, C]
        x = x.transpose(0, 1)
        return x
    
class TransformerDecoder(nn.Module):
    """Transformer-based decoder for image captioning."""
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer decoder layers
        decoder_layers = []
        for _ in range(num_layers):
            layer = nn.TransformerDecoderLayer(
                d_model,
                num_heads,
                dim_feedforward=d_model * 4,
                batch_first=True
            )
            decoder_layers.append(layer)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # Final classification layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: Optional[torch.Tensor] = None, hidden_state=None):
        # tgt: [B, T-1]
        # Project target to embedding space
        embedded = self.embedding(tgt)
        embedded = self.pos_encoding(embedded)  # [B, T-1, d_model]
        
        # Expand dimensions for attention
        embedded = embedded.unsqueeze(1)  # [B, 1, T-1, d_model]
        memory = memory.unsqueeze(1)  # [B, 1, 49, d_model]
        
        # Perform cross attention
        attn_output = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=None,
            memory_mask=None
        )
        
        # Project to vocabulary space
        output = self.fc_out(attn_output.squeeze(1))  # [B, T-1, vocab_size]
        return output, hidden_state

class PositionalEncoding(nn.Module):
    """Learned positional encoding to account for token order."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_encoding', self.encoding)

    def forward(self, x: torch.Tensor):
        x = x + self.pe_encoding[:x.size(1), :].unsqueeze(0)
        return x

def Conv2dNormActivation(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        dilation: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        inplace: bool = True,
):
    """Helper module that implements a Conv followed by BN and Activation."""
    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
    if activation_layer is None:
        activation_layer = nn.ReLU

    sequence = nn.Sequential()
    sequence.add_module(f'conv_{kernel_size}', nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias, groups
    ))
    if norm_layer:
        sequence.add_module(f'{norm_layer.__name__.split("_")[0]}_{out_channels}', norm_layer(out_channels))
    if activation_layer:
        sequence.add_module(f'act_{activation_layer.__name__}', activation_layer(inplace=inplace))
    return sequence