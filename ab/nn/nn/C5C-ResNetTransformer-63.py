import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional

class Net(nn.Module):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, prm: Dict, device: torch.device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.hidden_dim = 640
        
        # Encoder: CNN backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim)
        )
        
        # Decoder: LSTM with attention
        self.embed = nn.Embedding(out_shape[0], self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, out_shape[0])
        
        # Attention mechanism
        self.attention = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def train_setup(self, optimizer: optim.Optimizer, prm: Dict) -> None:
        """Set up the model for training."""
        for param in self.parameters():
            param.requires_grad = prm['trainable']
        if 'lr' in prm:
            for param_group in optimizer.param_groups:
                param_group['lr'] = prm['lr']
        if 'momentum' in prm:
            # Assuming SGD with momentum
            for param_group in optimizer.param_groups:
                param_group['momentum'] = prm['momentum']
    
    def learn(self, images: torch.Tensor, captions: torch.Tensor, teacher_forcing: bool = True) -> torch.Tensor:
        """Train the model on a batch of images and captions."""
        # Assertions
        assert images.dim() == 4, "Input images must be 4D tensors"
        assert captions.dim() == 2, "Captions must be 2D tensors"
        assert captions.size(1) == images.size(1) - 1, "Caption length must be image length - 1"
        
        # Move data to device
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        # Encode images
        memory = self.encoder(images)  # [B, 1, 640]
        
        # Initialize hidden state
        batch_size = images.size(0)
        hidden_state = None
        
        # Decoder steps
        outputs = []
        if teacher_forcing:
            # Start with SOS token
            captions_input = captions[:, :-1]
            for i in range(captions_input.size(1)):
                # Embed captions
                embedded = self.embed(captions_input[:, i])  # [B, 640]
                
                # Expand memory to match sequence length
                expanded_memory = memory.expand(-1, captions_input.size(1), -1)
                
                # Concatenate embedded and memory
                fused = torch.cat([embedded, expanded_memory[:, i]], dim=1)  # [B, 1280]
                
                # Apply attention
                attn_weights = self.softmax(self.attention(fused))  # [B, 1]
                
                # Weighted sum of memory
                context = torch.sum(attn_weights.unsqueeze(-1) * expanded_memory[:, i], dim=0)  # [B, 640]
                
                # Concatenate context with embedded
                fused_with_context = torch.cat([embedded, context], dim=1)  # [B, 1280]
                
                # LSTM step
                output, hidden_state = self.lstm(fused_with_context.unsqueeze(1), hidden_state)  # [B, 640]
                outputs.append(output)
            outputs = torch.cat(outputs, dim=1)  # [B, T, 640]
        else:
            # Without teacher forcing, start with SOS token
            captions_input = torch.full((batch_size,), 1, device=self.device)  # SOS token
            for i in range(captions_input.size(1)):
                embedded = self.embed(captions_input[:, i])
                expanded_memory = memory.expand(-1, captions_input.size(1), -1)
                fused = torch.cat([embedded, expanded_memory[:, i]], dim=1)
                attn_weights = self.softmax(self.attention(fused))
                context = torch.sum(attn_weights.unsqueeze(-1) * expanded_memory[:, i], dim=0)
                fused_with_context = torch.cat([embedded, context], dim=1)
                output, hidden_state = self.lstm(fused_with_context.unsqueeze(1), hidden_state)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=1)
        
        # Project outputs to vocabulary space
        logits = self.fc(outputs)  # [B, T, out_shape[0]]
        
        return logits
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Inference pass."""
        # Move data to device
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        # Encode images
        memory = self.encoder(images)
        
        # Initialize hidden state
        batch_size = images.size(0)
        hidden_state = None
        
        # Decoder steps
        outputs = []
        for i in range(captions.size(1)):
            embedded = self.embed(captions[:, i])
            expanded_memory = memory.expand(-1, captions.size(1), -1)
            fused = torch.cat([embedded, expanded_memory[:, i]], dim=1)
            attn_weights = self.softmax(self.attention(fused))
            context = torch.sum(attn_weights.unsqueeze(-1) * expanded_memory[:, i], dim=0)
            fused_with_context = torch.cat([embedded, context], dim=1)
            output, hidden_state = self.lstm(fused_with_context.unsqueeze(1), hidden_state)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        
        # Project outputs
        logits = self.fc(outputs)
        
        return logits

def supported_hyperparameters() -> Dict:
    return {'lr', 'momentum'}