import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        self.hidden_dim = 768  # >=640
        
        # Encoder: CNN backbone modified from classification model
        in_channels = int(in_shape[1])
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim)  # Projection to hidden_dim space
        )
        
        # Decoder: GRU based caption generator
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dropout = nn.Dropout(0.2)  # Structural tweak for regularization
        
    def init_zero_hidden(self, batch):
        # Initialize hidden state for decoder
        return torch.zeros(1, batch, self.hidden_dim).to(self.device)
    
    def train_setup(self, prm):
        # Setup training hyperparameters and optimizer
        self.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )
        self.criterion = self.criterion.to(self.device)
    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            # Assuming captions are in format [B, T] or [B, T, D]
            if captions.ndim == 3:
                captions = captions.reshape(-1, captions.shape[-1])
            
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Teacher forcing
            memory = self.encoder(images)
            embedded = self.embedding(captions)
            embedded = self.dropout(embedded)
            
            # Process captions with GRU
            logits, hidden_state = self.gru(embedded, None)
            logits = self.fc_out(logits)
            
            # Flatten logits and targets for loss calculation
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = captions.reshape(-1)
            
            # Calculate loss
            loss = self.criterion(logits_flat, targets_flat)
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
    
    def forward(self, images, captions=None, hidden_state=None):
        # Always encode the image
        memory = self.encoder(images).squeeze(1)  # [B, hidden_dim] for matching GRU
        
        if captions is not None:
            # Assume captions is [B, T]
            self.gru.flatten_parameters()  # Optimize GPU memory for batching
            embedded = self.embedding(captions)
            embedded = self.dropout(embedded)
            logits, hidden_state = self.gru(embedded, None)
            logits = self.fc_out(logits)
            return logits.permute(0, 2, 1), hidden_state
    
    # Additional methods could be added for inference