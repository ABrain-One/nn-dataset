import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.vocab_size = int(out_shape)
        
        # TODO: Replace self.encoder with custom encoder producing memory tensor [B, S, H] where H >= 640
        self.encoder = ViTEncoder(in_shape, hidden_dim=768, patch_size=16, num_heads=8, num_layers=12, dropout=0.1)
        
        # TODO: Replace self.rnn with custom decoder implementing forward(inputs, hidden_state, features) -> (logits, hidden_state)
        self.rnn = TransformerDecoder(vocab_size=self.vocab_size, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1)
        
        # For beam search
        self.scores = None
        self.done = None
        self.beam_size = 5
        
    def train_setup(self, hps):
        pass
        
    def learn(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        # Move images and captions to device
        images = images.to(self.device, dtype=torch.float32)
        if captions is not None:
            captions = captions.to(self.device, dtype=torch.long)
            
        # Get memory from encoder
        memory = self.encoder(images)
        
        # Teacher forcing
        if captions is not None:
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state, targets
            
        else:
            raise NotImplementedError()
            
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        # Move images and captions to device
        images = images.to(self.device, dtype=torch.float32)
        if captions is not None:
            captions = captions.to(self.device, dtype=torch.long)
            
        # Get memory from encoder
        memory = self.encoder(images)
        
        # If captions provided, do teacher forcing
        if captions is not None:
            inputs = captions[:, :-1]
            logits, hidden_state = self.rnn(inputs, hidden_state, memory)
            return logits, hidden_state
            
        # Else, do beam search
        else:
            # Initialize beam search
            self.scores = torch.zeros(images.size(0), self.beam_size, device=self.device)
            self.done = torch.zeros(images.size(0), self.beam_size, dtype=torch.bool, device=self.device)
            
            # Start with SOS token and create input for decoder
            input_seq = torch.full((images.size(0), self.beam_size, 1), 
                                  self.sos_index, 
                                  device=self.device, 
                                  dtype=torch.long)
            
            # Run decoder in autoregressive manner
            for step in range(self.max_length):
                # Repeat memory for each beam
                memory = memory.repeat(self.beam_size, 1, 1)
                
                # Repeat input_seq for each beam
                input_seq = input_seq.repeat(self.beam_size, 1, 1)
                
                # Get next token
                logits, hidden_state = self.rnn(input_seq, hidden_state, memory)
                
                # Convert to scores
                logits = logits[:, -1, :] / self.temperature
                logits = torch.log_softmax(logits, dim=-1)
                
                # Add scores
                self.scores = self.scores + logits
                
                # Get top k predictions and their indices
                self.scores_exp = self.scores.exp()
                top_k_scores, top_k_indices = self.scores_exp.topk(self.beam_size, dim=-1)
                
                # Renormalize beam scores
                self.scores = top_k_scores
                self.done = torch.logical_or(self.done, top_k_indices == self.eos_index)
                
                # Update input_seq
                input_seq = torch.cat([input_seq, top_k_indices.unsqueeze(1)], dim=1)
                
                # End if all sequences are done
                if self.done.all():
                    break
                    
            # Select best of beam
            top1 = self.scores.exp().unsqueeze(1).topk(self.beam_size, -1).indices
            best = top1[:, 0]
            return best
            
class ViTEncoder(nn.Module):
    def __init__(self, in_shape, hidden_dim=768, patch_size=16, num_heads=8, num_layers=12, dropout=0.1):
        super().__init__()
        self.in_shape = in_shape
        self.conv1 = nn.Conv2d(in_shape[1], hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.layernorm = nn.LayerNorm(hidden_dim, epsilon=1e-6)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)  # [B, hidden_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, hidden_dim, N]
        x = x.transpose(1, 2)  # [B, N, hidden_dim]
        x = self.layernorm(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, memory, hidden_state=None):
        embedded = self.embedding(tgt)  # [B, T, d_model]
        embedded = self.pos_encoder(embedded)  # [B, T, d_model]
        
        # Transpose memory to [S, B, d_model]
        memory = memory.transpose(0, 1)  # [S, B, d_model]
        
        # Create mask for decoder
        self_attn_mask = torch.triu(
            torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
        
        # Run transformer decoder
        out = self.transformer_decoder(tgt=embedded, memory=memory, self_attn_mask=self_attn_mask)
        
        # Final prediction
        out = self.fc_out(out)  # [B, T, vocab_size]
        return out, None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position.float() * div_term)
        self.encoding[:, 1::2] = torch.cos(position.float() * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        # x: [B, T, d_model]
        # Add positional encoding to the sequence
        seq_len = x.size(1)
        x = x + self.encoding[:, :seq_len]
        return self.dropout(x)

def supported_hyperparameters():
    return {'lr', 'momentum'}