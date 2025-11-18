import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        self.hidden_size = 768  # Must be >=640

        # Encoder: CNN without classification head
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Final projection to 768 channels
            nn.Conv2d(512, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )

        # Decoder: GRU with attention
        self.rnn = nn.GRU(
            input_size=768,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Define the attention mechanism
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(768, self.hidden_size)
        self.value_proj = nn.Linear(768, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size + 768, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size, out_shape[0])

        # Define the criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def init_zero_hidden(self, batch_size):
        # Initialize hidden state to zeros
        return torch.zeros(batch_size, self.hidden_size).to(self.device)

    def train_setup(self, prm):
        # Set up the model for training
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=prm['lr'])

    def learn(self, images, captions, prm):
        # Train the model on one batch
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        # Forward pass
        memory = self.encoder(images)  # [B, 1, 1, 768] (after global pool)
        memory = memory.squeeze(2)     # [B, 768]
        memory = memory.permute(1, 0)   # [768, B]
        
        # Initialize hidden state
        hidden_state = self.init_zero_hidden(captions.size(0))
        
        # Forward the decoder
        inputs = captions[:, :-1]  # [B, T-1]
        targets = captions[:, 1:]  # [B, T-1]
        
        # Pack the input sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, 
            torch.tensor([inputs.size(1)] * inputs.size(0), device=self.device),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Decoder forward pass
        outputs, hidden_state = self.rnn(packed_inputs, hidden_state)
        outputs = outputs[0]  # Unpack the output
        
        # Calculate the loss
        logits = self.fc(outputs)  # [B, T-1, vocab_size]
        loss = self.criterion(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, images, captions=None):
        # images: [B, 3, 224, 224] or [B, 3, 224, 224, ...] (batched)
        # captions: [B, T] or None

        # Forward the encoder
        memory = self.encoder(images)  # [B, 1, 1, 768]
        memory = memory.squeeze(2)     # [B, 768]
        memory = memory.permute(1, 0)   # [768, B]

        # If captions are provided, forward the decoder
        if captions is not None:
            captions = captions.to(self.device)
            # Pack the input sequence
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                captions[:, :-1], 
                torch.tensor([captions[:, :-1].size(1)] * captions.size(0), device=self.device),
                batch_first=True,
                enforce_sorted=False
            )
            # Decoder forward pass
            outputs, hidden_state = self.rnn(packed_inputs, self.init_zero_hidden(captions.size(0)))
            outputs = outputs[0]  # Unpack the output
            
            # Project the output to vocabulary space
            logits = self.fc(outputs)  # [B, T-1, vocab_size]
            
            return logits, hidden_state
        else:
            # Return the memory for inference
            return memory

def supported_hyperparameters():
    return {'lr','momentum'}