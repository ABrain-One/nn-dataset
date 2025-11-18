import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def supported_hyperparameters():
    return {'lr','momentum'}



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        pe = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pe * div_term)
        self.encoding[:, 1::2] = torch.cos(pe * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision_resnet50')  # Simplified loading
        
        # Modify to output higher dimensionality features
        in_channels = self.backbone.conv1.in_channels
        self.backbone.conv1 = nn.Conv2d(in_channels, 640, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.lstm_proj = nn.Linear(2048, 640)  # Projection for alignment
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = F.relu(features)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(3).squeeze(2)
        pooled = self.lstm_proj(pooled)
        pooled = pooled.unsqueeze(1)  # Shape [B, 1, 640]
        return pooled


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int = 640, vocab_size: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Projection layer to combine encoder features and decoder hidden state
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Condition on encoder memory
        expanded_memory = memory.expand(inputs.size(0), memory.size(0), memory.size(1))
        context = expanded_memory @ inputs.t() / math.sqrt(memory.size(1))
        context = F.softmax(context, dim=-1).transpose(0, 1)
        
        # Update hidden state
        if hidden_state is not None:
            projected = self.fc(hidden_state.squeeze(0))
            attended = context @ inputs
            combined = torch.cat((projected, attended), 1)
            updated_hidden = F.relu(self.combine(combined)).unsqueeze(0)
        else:
            updated_hidden = None
            
        # Pass through GRU and projection
        embedded = self.embeddings(inputs)
        gru_output, new_hidden = self.gru(embedded, updated_hidden)
        logits = self.fc(gru_output.contiguous().view(-1, self.hidden_size))
        return logits, new_hidden


class Net(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_shape: int, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape
        
        # Initialize encoder with required hidden dimension 640
        self.hidden_dim = 640
        self.encoder = CNNEncoder(pretrained=True).to(device)
        
        # Use GRU decoder (could alternatively use TransformerDecoder with batch_first)
        self.rnn = DecoderRNN(self.hidden_dim, self.vocab_size)
        
        # Ensure proper initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        # He initialization for convolution layers (handled by vision.pytorch's init)
        # Custom initialization for embedding and FC layers
        nn.init.uniform_(self.rnn.embeddings.weight, -0.1, 0.1)
        nn.init.zeros_(self.rnn.embeddings.bias)
        nn.init.uniform_(self.rnn.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.rnn.fc.bias)
        nn.init.uniform_(self.rnn.combine.weight, -0.01, 0.01)
        nn.init.zeros_(self.rnn.combine.bias)
        # He initializer equivalent (common for leaky ReLU)
        nn.init.kaiming_uniform_(self.rnn.gru.weight_ih_l0.data, a=0.01)
        nn.init.orthogonal_(self.rnn.gru.weight_hh_l0.data)
        
    def init_zero_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        # Initial hidden state for GRU
        return torch.zeros(1, batch, self.hidden_dim, device=device)
        
    def train_setup(self, prm: dict):
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions.to(self.device) if captions.dim() == 2 else captions[:, :, 0].to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            logits, _ = self.rnn(inputs, None, memory)
            
            loss = self.criterion(logits, targets.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()


if __name__ == '__main__':
    # Demo usage with random inputs
    demo_prm = {'lr': 1e-3, 'momentum': 0.9}
    net = Net((3, 224, 224), 10000, demo_prm, torch.device('cpu'))
    print(f"Encoder output shape: {net.encoder(torch.randn(8,3,224,224)).shape}")
    print(f"Decoder output shape with dummy input: {net.rnn(torch.randint(0,10000,(8, 10)), None, net.encoder(torch.randn(8,3,224,224)))}")