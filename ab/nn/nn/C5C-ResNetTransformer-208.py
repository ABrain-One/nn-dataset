import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, vocab_size, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = vocab_size
        self.hidden_dim = out_shape[-1]  # This is the dimension of the memory (at least 640)

        # Encoder section
        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(in_shape[1], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(256, self.hidden_dim)

        # Decoder section
        self.embedder = nn.Embedding(vocab_size, self.hidden_dim)
        self.rnn = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim, num_layers=1, batch_first=False)
        self.fc_out = nn.Linear(self.hidden_dim, vocab_size)

    def init_zero_hidden(self, batch_size, device):
        # Initialize hidden state for LSTM
        return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                torch.zeros(1, batch_size, self.hidden_dim).to(device))

    def train_setup(self, **kwargs):
        # Initialize the model with given hyperparameters
        pass

    def learn(self, images, captions):
        batch_size = images.size(0)
        device = images.device
        inputs = captions[:, :-1]   # [B, T-1]
        targets = captions[:, 1:]   # [B, T-1]

        # Forward pass
        memory = self.encoder(images)   # [B, 1, self.hidden_dim]
        hidden_state = self.init_zero_hidden(batch_size, device)
        logits, _ = self.rnn(inputs, hidden_state, memory)

        # Assertions
        assert images.dim() == 4
        assert logits.shape == (batch_size, inputs.shape[1], self.vocab_size)
        assert logits.shape[-1] == self.vocab_size

        # Loss computation
        loss = F.cross_entropy(logits, targets, ignore_index=0)
        return loss

    def forward(self, images, captions):
        # Forward pass during training
        memory = self.encoder(images)
        hidden_state = self.init_zero_hidden(captions.size(0), captions.device)
        logits, _ = self.rnn(captions, hidden_state, memory)
        return logits, hidden_state

    @staticmethod
    def supported_hyperparameters():
    return {'lr','momentum'}


    def encoder(self, x):
        x = self.encoder_backbone(x)
        x = self.global_pool(x).squeeze(3).squeeze(2)
        return self.project(x)

    def get_subsequent_mask(self, sz):
        """Get subsequent mask for the target sequence"""
        assert sz is not None
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.to(x.device)

    def decoder(self, inputs, hidden_state, memory):
        # Expand memory to match sequence length
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        expanded_memory = memory.expand(batch_size, seq_length, -1)
        
        # Embed inputs
        embedded = self.embedder(inputs)
        
        # Concatenate embedded inputs and expanded memory
        inputs_concat = torch.cat([embedded, expanded_memory], -1)
        
        # Process through RNN
        output, hidden_state = self.rnn(inputs_concat, hidden_state)
        
        # Project output to vocabulary space
        logits = self.fc_out(output)
        return logits, hidden_state

    def __call__(self, *args, **kwargs):
        if len(args) == 2:
            # This is the training mode: images and captions
            return self.learn(*args, **kwargs)
        elif len(args) == 1 and len(kwargs) == 1 and 'captions' in kwargs:
            # This is the inference mode: only images
            return self.forward(args[0], kwargs['captions'])
        else:
            # This is the general forward pass
            return self.forward(*args, **kwargs)