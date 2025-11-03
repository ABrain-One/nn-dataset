import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Encoder(nn.Module):
    def __init__(self, in_shape, hidden_size):
        super().__init__()
        self.in_shape = in_shape
        self.hidden_size = hidden_size
        # CNN architecture definition
        self.cnn = nn.Sequential(
            nn.Conv2d(in_shape[1], hidden_size, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # ... rest of the CNN architecture
        )
    
    def forward(self, images):
        # images: [B, C, H, W]
        # features: [B, L, hidden_size]
        return self.cnn(images)

class Decoder(nn.Module):
    def __init__(self, out_shape, hidden_size, num_heads):
        super().__init__()
        self.out_shape = out_shape
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # TransformerDecoder architecture definition
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, out_shape[0])
    
    def forward(self, captions, features):
        # captions: [B, T]
        # features: [B, L, hidden_size]
        # outputs: [B, T-1, out_shape]
        # hidden_state: [B, num_layers, num_heads, hidden_size]
        memory = features[:,0,:]  # [B, hidden_size]
        outputs = self.decoder(captions, memory)
        logits = self.classifier(outputs)
        return logits, outputs

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device, *_, **__):
        
            # ---- API aliases (auto-injected) ----
            self.in_shape = in_shape
            self.out_shape = out_shape
            self.device = device
            self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
            self.vocab_size = out_shape[0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
            self.out_dim = self.vocab_size
            self.num_classes = self.vocab_size

            # Backward-compat local aliases (old LLM patterns)
            vocab_size = self.vocab_size
            out_dim = self.vocab_size
            num_classes = self.vocab_size
            in_channels = self.in_channels
super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.prm = prm
        
        self.in_channels = in_shape[1] if isinstance(in_shape, (tuple, list)) else 3
        self.vocab_size = out_shape[0][0][0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
        self.out_dim = self.vocab_size
        self.num_classes = self.vocab_size
        
        # Backward-compat local aliases (old LLM patterns)
        self.sos_idx = self.vocab_size - 1  # SOS token index
        
        # Define encoder and decoder
        self.encoder = Encoder(in_shape, self.vocab_size)
        self.decoder = Decoder(self.vocab_size, self.vocab_size, num_heads=8)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    def learn(self, train_data):
        # Training loop implementation
        self.train()
        for batch in train_data:
            images, captions = batch
            features = self.encoder(images)
            captions_for_decoding = captions.clone()
            if captions_for_decoding.size(1) > 0:
                sos_vector = torch.full((images.size(0), 1), self.sos_idx, 
                                        dtype=torch.long, device=images.device)
                captions_for_decoding = torch.cat([sos_vector, captions_for_decoding], dim=1)
            
            outputs, _ = self.decoder(captions_for_decoding, features)
            targets = captions[:,1:].contiguous()
            loss = self.criteria[0](
                outputs.contiguous().view(-1, outputs.shape[2]),
                targets.contiguous().view(-1)
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def forward(self, images, captions=None, hidden_state=None):
        # images: [B, C, H, W]
        # captions: None or [B, T] or [B, T, V] (then convert to [B, T])
        # hidden_state: [B, num_layers, num_heads, hidden_size] (optional)

        batch_size = images.size(0)
        if captions is not None and captions.ndim == 3:
            captions = captions[:,0,:]  # [B, T]

        if captions is not None:
            # Training: captions are [B, T]
            sos_idx = self.sos_idx
            if captions.size(1) > 0:
                sos_vector = torch.full((batch_size, 1), sos_idx, 
                                        dtype=torch.long, device=images.device)
                captions_for_decoding = torch.cat([sos_vector, captions], dim=1)
            else:
                captions_for_decoding = captions

            outputs, hidden_state = self.decoder(captions_for_decoding, self.encoder(images))
            return outputs, hidden_state
        else:
            # Inference: generate captions
            self.eval()
            memory = self.encoder(images)[:,0,:]  # [B, hidden_size]
            captions = torch.full((batch_size, 1), self.sos_idx, 
                                 dtype=torch.long, device=images.device)
            generated_captions = []
            for _ in range(self.prm['max_length']):
                if hidden_state is not None:
                    decoder_output, hidden_state = self.decoder(captions, memory)
                else:
                    decoder_output = self.decoder(captions, memory)
                
                logits = self.classifier(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                
                # Check if we've reached EOS
                eos_mask = (next_token == self.vocab_size - 1).squeeze(1)
                
                # Append next token and break if EOS is reached
                captions = torch.cat([captions, next_token], dim=1)
                if torch.any(eos_mask):
                    break
            
            return captions, hidden_state

def supported_hyperparameters():
    return {'lr', 'momentum'}