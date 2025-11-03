import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

class SEBlock(nn.Module):
    def __init__(self, channel: int, ratio: int = 4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super(InvertedResidual, self).__init__()
        self.identity = stride == 1 and inp == oup
        
        self.conv = nn.Sequential(
            # Expansion
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(inp * expand_ratio, oup, 3, stride, 1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            # Projection
            nn.Conv2d(oup, oup, 1, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

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
        self.device = device
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = out_shape[0][0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        
        # Encoder: ResNet-like with SE blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            # Stage 1: 2x2 stride
            self._make_layer(64, 256, 3, 2),
            
            # Stage 2: 1x1 stride
            self._make_layer(256, 512, 3, 1),
            
            # Stage 3: 1x1 stride
            self._make_layer(512, 1024, 2, 1),
            
            # Stage 4: 1x1 stride
            self._make_layer(1024, 1280, 2, 1),
            
            # Final stage
            nn.Conv2d(1280, 768, 1, 1, bias=False),
            nn.BatchNorm2d(768),
            SEBlock(768)
        )
        
        # Decoder: LSTM-based with attention
        self.decoder = nn.LSTM(
            input_size=768,
            hidden_size=768,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )
        
        self.embedding = nn.Embedding(self.vocab_size, 768)
        self.attention = nn.Linear(768, 768)
        self.fc = nn.Linear(768, self.vocab_size)
        self.device = device
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        _layer_list = []
        for i in range(blocks):
            if i == 0 and stride != 1:
                _layer_list.append(InvertedResidual(in_channels, out_channels, stride, 6))
            else:
                _layer_list.append(InvertedResidual(in_channels, out_channels, 1, 6))
            in_channels = out_channels
        return nn.Sequential(*_layer_list)
    
    def init_zero_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, batch_size, 768).to(self.device),
                torch.zeros(1, batch_size, 768).to(self.device))
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )

    
    def learn(self, train_data: List[Tuple[torch.Tensor, List[str]]]):
        self.train_setup({'lr': 0.001})
        for images, captions in train_data:
            features = self.encoder(images)
            embedded = self.embedding(captions[:-1])
            embedded = embedded.view(embedded.size(0), embedded.size(1), 768)
            
            h0, c0 = self.init_zero_hidden(images.size(0))
            outputs, (h_n, c_n) = self.decoder(embedded, (h0, c0))
            
            loss = self.criteria[0](outputs.view(-1, self.vocab_size), captions[1:].view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # Get encoder features
        features = self.encoder(images)
        
        # If captions is provided, we are in teacher forcing mode
        if captions is not None:
            # Embed the captions
            embedded = self.embedding(captions)
            
            # If hidden_state is provided, use it; otherwise initialize
            if hidden_state is None:
                h0, c0 = self.init_zero_hidden(images.size(0))
            else:
                h0, c0 = hidden_state
                
            # Pass through the decoder
            outputs, hidden_state = self.decoder(embedded, (h0, c0))
            
            # Apply attention to features
            attention_weights = F.softmax(self.attention(outputs[:, -1, :]), dim=1)
            context = torch.sum(outputs[:, -1, :] * attention_weights, dim=1)
            
            # Project to vocabulary distribution
            logits = self.fc(context)
            
            return logits, hidden_state
        
        # Otherwise, we are in inference mode
        else:
            # Initialize hidden state
            h0, c0 = self.init_zero_hidden(images.size(0))
            
            # Start with SOS token
            captions = [self.vocab_size - 1]  # SOS index
            
            # Generate captions until EOS
            while len(captions) < 50:  # max length
                embedded = self.embedding(captions)
                embedded = embedded.view(len(captions), 1, 768)
                
                outputs, (h0, c0) = self.decoder(embedded, (h0, c0))
                
                # Get the last output's distribution
                last_output = outputs[-1, -1, :]
                last_output = self.fc(last_output)
                
                # Sample next token
                probs = F.softmax(last_output, dim=0)
                next_token = torch.multinomial(probs, 1)
                captions.append(next_token.item())
                
                # If next_token is EOS, break
                if next_token.item() == self.vocab_size - 2:  # EOS index
                    break
            
            return captions

def supported_hyperparameters():
    return {'lr', 'momentum'}