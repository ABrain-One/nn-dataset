import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=768):
        super().__init__()
        self.d_model = hidden_dim
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense Blocks
        block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        block5 = nn.Sequential(
            nn.Conv2d(512, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Create dense blocks with transition layers
        self.dense1 = self._make_dense_layer(64, 64, 1)
        self.dense2 = self._make_dense_layer(128, 128, 2)
        self.dense3 = self._make_dense_layer(256, 256, 2)
        self.dense4 = self._make_dense_layer(512, 512, 2)
        self.dense5 = self._make_dense_layer(768, 768, 1)
        
        # Projection layer
        self.projection = nn.Conv2d(768, self.d_model, kernel_size=1)
    
    def _make_dense_layer(self, input_channels, output_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))
            input_channels += output_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.projection(F.adaptive_avg_pool2d(x, (1, 1))).flatten(1)
        return x.unsqueeze(1)

class CustomTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer decoder layers
        decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, batch_first=True)
            decoder_layers.append(layer)
        self.transformer_dec = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.hidden_state = None

    def forward(self, inputs, hidden_state=None, memory=None):
        # Normalize memory and inputs
        normalized_memory = (memory - memory.min()) / (memory.max() - memory.min())
        
        # Apply positional encoding
        embedded = self.embedding(inputs)
        embedded = self.pos_encoding(embedded)
        
        # Set sequence length based on memory
        if memory is not None:
            seq_len = memory.size(1)
        else:
            seq_len = inputs.size(1)
        
        # Create lookahead mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=inputs.device), diagonal=1)
        
        # Decode
        output = self.transformer_dec(embedded, normalized_memory, tgt_mask=mask)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        # Update hidden_state
        self.hidden_state = hidden_state if hidden_state is not None else None
        
        return logits, self.hidden_state

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding tensor (buffer)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])
        
        # Get input channels
        in_channels = int(in_shape[1])
        
        # Create encoder and decoder
        self.encoder = CNNEncoder(in_channels=in_channels, hidden_dim=768)
        self.decoder = CustomTransformerDecoder(vocab_size=self.vocab_size, d_model=768, num_layers=6, nhead=8)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = None

    def init_zero_hidden(self, batch: int, device: torch.device):
        return None, None

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        lr = max(float(prm.get('lr', 1e-3)), 3e-4)
        beta1 = min(max(float(prm.get('momentum', 0.9)), 0.7), 0.99)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:] if captions.ndim == 3 else captions
            caps = caps.long().to(self.device)
            
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encoder(images)
            logits, _ = self.decoder(inputs, None, memory)
            
            loss = self.criterion(logits.reshape(-1, self.decoder.d_model).double(), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encoder(images)
        
        if captions is not None:
            if captions.ndim == 2:
                captions = captions.unsqueeze(1)  # Add sequence dimension
                
            caps_input = captions[:, 0, :] if captions.ndim == 3 else captions
            caps_input = caps_input.long()
            outputs, new_hidden = self.decoder(caps_input, hidden_state, memory)
            
            assert outputs.shape == (images.size(0), captions.size(1)-1, self.vocab_size)
            assert new_hidden.shape == (images.size(0), hidden_state.shape[1] if hidden_state is not None else 0)
            
            return outputs.double(), new_hidden
            
        else:
            return None, None