import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden_dim=640):
        super().__init__()
        self.hidden_dim = hidden_dim
        # TODO: Replace self.encoder with custom encoder producing memory tensor [B, S, H] where H >= 640
        self.encoder = nn.Identity()
        self.rnn = nn.RNN(input_size=encoder_output_size, hidden_size=hidden_dim, batch_first=False)

    def train_setup(self, ...):
        # This method should handle training setup like device configuration, etc.
        pass

    def learn(self, images, captions):
        # Process captions and images to produce inputs and targets
        if captions.ndim == 3:
            # Flatten first dimension if needed
            captions = captions.reshape(-1, captions.shape[1])
        inputs = captions[:, :-1]
        targets = captions[:, 1:]
        memory = self.encoder(images)
        return memory, inputs, targets

    def forward(self, images, captions=None, hidden_state=None):
        # Get memory from encoder
        memory = self.encoder(images)
        
        # If captions are provided, process them
        if captions is not None:
            # Convert captions to appropriate format
            if captions.ndim == 3:
                caps = captions[:, 0, :].long().to(self.device)
            else:
                caps = captions.long().to(self.device)
            inputs = caps[:, :-1]
            
            # Get logits from RNN
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            
            # Shape assert for logits
            assert logits.dim() == 3, "Logits must be 3D"
            assert logits.shape[2] == self.hidden_dim, "Hidden dimension mismatch"
            
            # Return logits and hidden_state
            return logits, hidden_state
        
        # Return None if captions are not provided
        return None, None

def supported_hyperparameters():
    return {'lr','momentum'}