import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        # Define encoder and decoder here
        self.encoder = EncoderCNN()
        self.decoder = DecoderTransformer(vocab_size=1000)  # Example vocab size

    @staticmethod
    def supported_hyperparameters():
    return {'lr','momentum'}


    def train_setup(self, hps):
        self.optimizer = optim.Adam(self.parameters(), lr=hps['lr'])

    def learn(self, inputs, hidden_state, memory, targets=None):
        # Training step
        self.optimizer.zero_grad()
        logits, hidden_state = self.rnn(inputs, hidden_state, memory)
        if targets is not None:
            loss = self.calculate_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
        return loss, logits, hidden_state

    def forward(self, inputs, hidden_state, memory):
        # Forward pass for teacher forcing
        logits, hidden_state = self.rnn(inputs, hidden_state, memory)
        return logits, hidden_state

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(512, 1024)  # Output dimension >=640

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size):
        super(DecoderTransformer, self).__init__()
        self.d_model = 1024
        self.num_layers = 6
        self.nhead = 8
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.fc_out = nn.Linear(self.d_model, vocab_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

    def forward(self, inputs, hidden_state, memory):
        # inputs: [B, T-1]
        # hidden_state: unused
        # memory: [B, 1, 1024]
        
        embedded = self.embedding(inputs)
        seq_len = inputs.size(1)
        tgt_mask = self._generate_square_attention_mask(seq_len)
        
        out = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits, None

    def _generate_square_attention_mask(self, seq_length):
        # Create a mask to hide future positions
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).type(torch.bool)
        return mask

# --- auto-closed by AlterCaptionNN ---