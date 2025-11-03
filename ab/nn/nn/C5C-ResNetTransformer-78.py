import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Encoder: CNN that outputs a single vector per image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Linear(768, 768)
        )
        # Decoder: LSTM
        self.embedding = nn.Embedding(args['vocab_size'], 768)
        self.lstm = nn.LSTM(input_size=768 + 768, hidden_size=768, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(768, args['vocab_size'])
        
    def train_setup(self, args):
        self.lstm.train()
        
    def learn(self, args, images, captions):
        # Teacher forcing training
        memory = self.encoder(images)
        memory = memory.unsqueeze(1)  # [B, 1, 768]
        
        # Embed the input captions
        embedded = self.embedding(captions[:, :-1])  # [B, T-1, 768]
        
        # Expand memory to match the sequence length
        memory = memory.expand(-1, embedded.size(1), -1)  # [B, T-1, 768]
        
        # Concatenate embedded and memory
        inputs = torch.cat([embedded, memory], dim=2)  # [B, T-1, 1536]
        
        # Initialize hidden state if not provided
        if 'hidden_state' not in args:
            hidden_state = None
        else:
            hidden_state = args['hidden_state']
            
        # Forward pass through LSTM
        output, hidden_state = self.lstm(inputs, hidden_state)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        # Calculate loss
        loss = F.cross_entropy(logits, captions[:, 1:], reduction='mean')
        
        return loss, logits, hidden_state
    
    def forward(self, args, images, captions=None, hidden_state=None):
        # Get memory from encoder
        memory = self.encoder(images)
        memory = memory.unsqueeze(1)  # [B, 1, 768]
        
        # If captions are provided, use teacher forcing
        if captions is not None:
            # Embed the input captions
            embedded = self.embedding(captions[:, :-1])  # [B, T-1, 768]
            
            # Expand memory to match the sequence length
            memory = memory.expand(-1, embedded.size(1), -1)  # [B, T-1, 768]
            
            # Concatenate embedded and memory
            inputs = torch.cat([embedded, memory], dim=2)  # [B, T-1, 1536]
            
            # Forward pass through LSTM
            output, hidden_state = self.lstm(inputs, hidden_state)
            
            # Project to vocabulary
            logits = self.fc_out(output)
            
            return logits, hidden_state
        
        # Otherwise, generate captions autoregressively
        else:
            # Initialize hidden state if not provided
            if hidden_state is None:
                hidden_state = None
                
            # Start with a start token (index 0)
            input = torch.tensor([0]).expand(images.size(0), 1)
            embedded = self.embedding(input)
            memory = memory.expand(-1, embedded.size(1), -1)
            inputs = torch.cat([embedded, memory], dim=2)
            
            # Forward pass through LSTM
            output, hidden_state = self.lstm(inputs, hidden_state)
            
            # Project to vocabulary
            logits = self.fc_out(output)
            
            return logits, hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}