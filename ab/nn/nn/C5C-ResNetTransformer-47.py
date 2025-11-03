import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # Encoder: modified ResNet50 backbone
        self.encoder = nn.Sequential(
            # Define the encoder layers here
            # Example: 
            nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * (in_shape[1]//2) * (in_shape[2]//2), 768)
        )
        
        # Decoder: Transformer decoder
        self.embedding_size = 768
        self.hidden_size = 768
        self.num_layers = 6
        self.nhead = 8
        
        self.embedding = nn.Embedding(out_shape, self.embedding_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, self.embedding_size))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.hidden_size, self.nhead, batch_first=True),
            self.num_layers
        )
        self.generator = nn.Linear(self.hidden_size, out_shape)
        
        # Initialize hidden_state for the decoder
        self.init_zero_hidden = lambda batch: (torch.zeros(1, batch, self.hidden_size, device=self.device), None)

    def train_setup(self, prm):
        # Set up training components here
        pass

    def learn(self, batch, captions, images, prm):
        # Implement training logic here
        pass

    def forward(self, images, captions=None, hidden_state=None):
        # images: [B, C, H, W]
        # captions: [B, T] if provided, else None

        if captions is not None:
            # Use teacher forcing
            memory = self.encoder(images)
            # The decoder's forward expects: inputs, hidden_state, memory
            # We'll use the captions (without the last token) as inputs
            inputs = captions[:, :-1]
            embedded = self.embedding(inputs)
            embedded = embedded.permute(1, 0, 2)  # [T, B, E]
            memory = memory.permute(1, 0, 2)      # [S, B, E]
            
            # Add positional encoding
            embedded = embedded + self.pos_encoder
            
            # Run transformer decoder
            output = self.transformer(embedded, memory)
            logits = self.generator(output)
            
            # The API expects the output to be (logits, hidden_state)
            return logits, hidden_state
        
        # If captions is None, we are to generate captions
        # We'll use the initial hidden_state from the decoder
        memory = self.encoder(images)
        # Generate captions using beam search or other method
        # This is a placeholder for the generation logic
        return None, None

def supported_hyperparameters():
    return {'lr','momentum'}