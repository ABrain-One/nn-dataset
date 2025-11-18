import torch
import torch.nn as nn
from typing import Optional, Tuple

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = out_shape[0]
        self.seq_length = out_shape[0]
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.output_projection = nn.Linear(768, self.vocab_size)
    
    def build_encoder(self):
        # Encoder: A CNN that outputs a single vector of size 768
        in_channels = self.in_shape[1]
        encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        return encoder
    
    def build_decoder(self):
        # Decoder: Transformer decoder
        d_model = 768
        nhead = 8
        num_layers = 2
        dim_feedforward = 2048
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        return decoder
    
    def train_setup(self, **kwargs):
        # Set up the model for training
        self.to(kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.get('lr', 0.001))
        self.loss_fn = nn.CrossEntropyLoss()
    
    def learn(self, epochs=10, **kwargs):
        # Train the model for the specified number of epochs
        device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for epoch in range(epochs):
            for inputs, targets in kwargs.get('train_loader', []):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self(inputs, targets)
                loss = self.loss_fn(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1} completed")
    
    def forward(self, images, captions):
        # images: (batch, channels, height, width)
        # captions: (batch, seq_length)
        # Teacher forcing: Pass the previous token in as the next input
        
        # Encoder
        encoder_output = self.encoder(images)  # (batch, 512, 1, 1)
        encoder_output = encoder_output.reshape(encoder_output.size(0), -1)  # (batch, 512)
        
        # Project encoder output to decoder dimension
        memory = encoder_output.unsqueeze(1)  # (batch, 1, 512)
        memory = self.output_projection(memory)  # (batch, 1, 768)
        
        # Decoder
        decoder_output = self.decoder(memory, captions)  # (batch, seq_length, 768)
        
        # Project to vocabulary space
        logits = self.output_projection(decoder_output)  # (batch, seq_length, vocab_size)
        
        # Shape assert
        assert logits.shape == (images.size(0), captions.size(1), self.vocab_size), \
            f"Expected logits shape (batch, seq_length, vocab_size) but got {logits.shape}"
        
        return logits, memory

    @staticmethod
    def supported_hyperparameters():
    return {'lr','momentum'}



# Dummy implementation for beam search
def beam_search(images, net, beam_size=5):
    pass

# Example usage
if __name__ == '__main__':
    # Create a dummy model
    model = Net(in_shape=(3, 224, 224), out_shape=(10000, 15))
    # Dummy input
    images = torch.randn(1, 3, 224, 224)
    captions = torch.randint(0, 10000, (1, 15))
    # Forward pass
    logits, hidden_state = model(images, captions)
    print(logits.shape)  # Should be [1, 14, 10000]