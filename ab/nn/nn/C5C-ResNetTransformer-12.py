import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        in_channels = int(in_shape[1])
        
        # Encoder architecture based on ResNet-50 principles
        # Define encoder feature extractor
        self.encoder = nn.Sequential(
            # Stem
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Stage 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Stage 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Stage 3
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            # Stage 4
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            
            # Final spatial reduction
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Set hidden dimension (must be ≥640)
        self.hidden_dim = 768
        
        # Decoder architecture using TransformerDecoder
        d_model = self.hidden_dim
        num_layers = 6
        nhead = 8
        
        # Calculate the proper dimensions for the projection
        # Each block contributes differently to the output dimension
        # Stage 1: 64 -> 256 (projection needed)
        # Stage 2: 256 -> 512 (projection needed)
        # Stage 3: 512 -> 1024 (projection needed)
        # Stage 4: 1024 -> 2048 (projection needed)
        
        # Projection layers for each stage output
        self.stage1_proj = nn.Linear(64, d_model)
        self.stage2_proj = nn.Linear(256, d_model)
        self.stage3_proj = nn.Linear(512, d_model)
        self.stage4_proj = nn.Linear(1024, d_model)
        
        # Combine projections from all stages to get final memory features [B, S, H]
        # S = sequence length, H = hidden dimension (768)
        # Instead of pooling to 1×1, we'll output all stages sequentially
        self.concat_stages = nn.Sequential()
        
        # Add the projection layers to concat_stages as sequential operations
        self.concat_stages.add_module('stage1', self.stage1_proj)
        self.concat_stages.add_module('stage2', self.stage2_proj)
        self.concat_stages.add_module('stage3', self.stage3_proj)
        self.concat_stages.add_module('stage4', self.stage4_proj)
        
        # Transformer decoder setup
        encoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        
        # Embedding layer for tokens
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Final linear projection for output tokens
        self.fc_out = nn.Linear(d_model, self.vocab_size)

        # Initialize decoder weights
        self._reset_parameters()

        # Save important hyperparameters
        self.learning_rate = prm.get('lr', 1e-3)
        self.momentum = prm.get('momentum', 0.9)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_zero_hidden(self, batch, device):
        # Transformer decoder doesn't require explicit hidden states initialization
        return None

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:, 0, :].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            memory = self.encode_images(images)
            logits, _ = self.decode_sequence(inputs, memory)
            
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def encode_images(self, images):
        """Extract hierarchical features from images using the encoder backbone"""
        encoded = self.encoder(images)
        # Flatten the spatial representations from each stage and project
        # There are four main stages before the final pooling
        s1 = F.adaptive_avg_pool2d(encoded, (7, 7))[:, :, 0, 0]  # Stage 1 pooled: [B, 64]
        s2 = encoded[:, :, 7:9, 7:9]  # Stage 2: approximately, depending on actual layers
        s3 = encoded[:, :, 14:17, 14:17]  # Stage 3: approximately
        s4 = encoded[:, :, 28:, 28:]  # Stage 4: approximate location
        
        # Apply projection to all features
        s1 = self.concat_stages[0](s1)  # Project to 768
        s2 = self.concat_stages[1](s2.reshape(s2.size(0), -1))  # View and project to 768
        s3 = self.concat_stages[2](s3.reshape(s3.size(0), -1))  # View and project to 768
        s4 = self.concat_stages[3](s4.reshape(s4.size(0), -1))  # View and project to 768
        
        # Concatenate all features to form the memory tensor [B, S_total, H]
        memory = torch.cat([s1, s2, s3, s4], dim=1)
        return memory

    def decode_sequence(self, inputs, memory):
        """Decode target sequences using transformer decoder"""
        # Memory has shape [B, S_total, H] (S_total is sum of features from each stage)
        # Inputs: target sequences [B, T_in]
        
        # Get device info
        device = inputs.device
        
        # Embed input tokens and apply positional encoding
        embedded = self.embedding(inputs)  # [B, T_in, H]
        embedded = self.pos_encoding(embedded)  # [B, T_in, H]
        
        # Run through transformer decoder
        output = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            self_attention_mask=None,
            encoder_decoder_mask=None
        )
        
        # Final prediction
        logits = self.fc_out(output)  # [B, T_in, vocab_size]
        return logits, None

    def forward(self, images, captions=None, hidden_state=None):
        """Main forward pass for image captioning"""
        images = images.to(self.device, dtype=torch.float32)
        memory = self.encode_images(images)
        
        if captions is not None:
            # For teacher forcing
            embedded_inputs = self.embedding(captions)
            encoded_inputs = self.pos_encoding(embedded_inputs)
            output, _ = self.transformer_decoder(encoded_inputs, memory)
            logits = self.fc_out(output)
            
            # Return shape [B, T_out-1, vocab_size]
            assert logits.shape == (*encoded_inputs.shape[:2], self.vocab_size)
            return logits, None
        
        else:
            # For generation (should be integrated with BeamSearch later)
            raise NotImplementedError("Generation with captions None not implemented")

    def generate_caption(self, image, max_length=20, sos_index=0, eos_index=1, beam_width=4):
        # NOT IMPLEMENTED AS PER REQUIREMENT BUT STRUCTURED FOR COMPLETION IN NEXT STEP
        pass

class PositionalEncoding(nn.Module):
    """Learnable positional encoding module"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).requires_grad_(False).to(torch.float32)

    def forward(self, x):
        # x: [..., L, d_model] where L ≤ max_len
        seq_len = x.size(-1)
        x = x + self.encoding[:, :, :seq_len]
        return self.dropout(x)

def supported_hyperparameters():
    return {'lr','momentum'}
