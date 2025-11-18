import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.vocab_size = out_shape[0]
        self.hidden_dim = 768  # >=640

        # Build encoder
        self.encoder = self.build_encoder(in_shape[1])
        
        # Build decoder
        self.decoder = self.build_decoder(self.vocab_size)

    def build_encoder(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim)
        )

    def build_decoder(self, vocab_size):
        return nn.LSTM(
            input_size=512,  # This is the feature size from the encoder
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        ).to(self.device)

    def init_zero_hidden(self, batch, device):
        return torch.zeros((2, batch, self.hidden_dim), device=device)

    def train_setup(self, optimizer, scheduler, **prm):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = prm['lr']
        self.momentum = prm['momentum']

    def learn(self, images, captions):
        # Convert images to device
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        # Forward pass
        memory = self.encoder(images)  # [B, 768]
        memory = memory.unsqueeze(1)  # [B, 1, 768]
        
        # Decoder setup
        batch_size = images.size(0)
        seq_length = captions.size(1)
        hidden_state = self.init_zero_hidden(batch_size, self.device)
        
        # Decoder forward pass
        outputs, _ = self.decoder(captions, hidden_state, memory)  # [B, seq_length, hidden_dim]
        
        # Project outputs to vocabulary
        logits = torch.zeros((batch_size, seq_length, self.vocab_size), device=self.device)
        for t in range(seq_length):
            # Get the embedding for the t-th token
            token_embedding = self.token_embedding(captions[:, t])
            # Project through the output layer
            logit_t = self.output_projection(token_embedding)
            logits[:, t] = logit_t
            
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, self.vocab_size), captions.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def forward(self, images, captions=None):
        # Convert images to device
        images = images.to(self.device)
        if captions is not None:
            captions = captions.to(self.device)
            
        # Forward pass through encoder
        memory = self.encoder(images)  # [B, 768]
        memory = memory.unsqueeze(1)  # [B, 1, 768]
        
        # Decoder setup
        batch_size = images.size(0)
        if captions is not None:
            seq_length = captions.size(1)
            # Decoder forward pass
            outputs, hidden_state = self.decoder(captions, None, memory)  # [B, seq_length, hidden_dim]
            
            # Project outputs to vocabulary
            logits = torch.zeros((batch_size, seq_length, self.vocab_size), device=self.device)
            for t in range(seq_length):
                # Get the embedding for the t-th token
                token_embedding = self.token_embedding(captions[:, t])
                # Project through the output layer
                logit_t = self.output_projection(token_embedding)
                logits[:, t] = logit_t
                
            return logits, hidden_state
        else:
            # Inference mode
            # Start with <SOS> token
            sos_index = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
            captions = sos_index
            
            # Decoder forward pass
            outputs, hidden_state = self.decoder(captions, None, memory)
            
            # Project outputs to vocabulary
            logits = torch.zeros((batch_size, captions.shape[1], self.vocab_size), device=self.device)
            for t in range(captions.shape[1]):
                # Get the embedding for the t-th token
                token_embedding = self.token_embedding(captions[:, t])
                # Project through the output layer
                logit_t = self.output_projection(token_embedding)
                logits[:, t] = logit_t
                
            return logits, hidden_state

    def token_embedding(self, tokens):
        # This is a placeholder for the token embedding layer
        # In a real implementation, this would be defined in the decoder
        return tokens

    def output_projection(self, hidden_state):
        # This is a placeholder for the output projection layer
        # In a real implementation, this would be defined in the decoder
        return hidden_state

def supported_hyperparameters():
    return {'lr','momentum'}