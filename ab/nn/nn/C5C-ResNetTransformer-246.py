import math
import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        
        # Encoder configuration
        enc_in_channels = in_shape[1]
        enc_mid_channels = 64
        enc_final_channels = 640
        
        # Decoder configuration
        dec_in_features = enc_final_channels
        dec_hidden_size = enc_final_channels
        dec_num_layers = 6
        dec_num_heads = 8
        dec_vocab_size = out_shape[0]
        
        # Build encoder
        self.stem = nn.Conv2d(enc_in_channels, enc_mid_channels, kernel_size=7, stride=2, padding=3)
        self.stem_bn = nn.BatchNorm2d(enc_mid_channels)
        self.stage1 = self.make_stage(3, 64, 0.1)
        self.stage2 = self.make_stage(6, 128, 0.1)
        self.stage3 = self.make_stage(12, 256, 0.1)
        self.stage4 = self.make_stage(24, 512, 0.1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(512, enc_final_channels)
        
        # Build decoder
        self.token_embedding = nn.Embedding(dec_vocab_size, dec_in_features)
        self.pos_encoder = PositionalEncoding(dec_in_features)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dec_in_features, nhead=dec_num_heads, 
                                               dim_feedforward=2048, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_num_layers)
        self.output_proj = nn.Linear(dec_in_features, dec_vocab_size)
        
        # Initialize hyperparameters
        self.hidden_dim = enc_final_channels
        self.embed_dim = dec_in_features
        self.num_heads = dec_num_heads
        
        # Optimizer components
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
    def make_stage(self, repeats: int, mid_channels: int, dropout_p: float):
        layers = []
        for _ in range(repeats):
            layer = nn.Sequential(
                nn.Conv2d(512, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p)
            )
            layers.append(layer)
        return nn.Sequential(*layers)
    
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None):
        if captions is None:
            raise NotImplementedError()
            
        images_float = images.float().to(self.device)
        memory = self.extract_features(images_float)
        inputs = captions[:, :-1]
        
        # Project inputs if needed
        if inputs.dim() == 3:
            embedded_inputs = inputs[:, :, :]   # Already embedded
        else:
            embedded_inputs = self.token_embedding(inputs)
            
        embedded_inputs = self.pos_encoder(embedded_inputs)
        outputs = self.transformer_decoder(tgt=embedded_inputs, memory=memory)
        
        # Final projection
        logits = self.output_proj(outputs)
        hidden_state = self.transformer_decoder.layers[-1].self_attn.self_k.value
        
        return logits, hidden_state
    
    def extract_features(self, images):
        x = F.relu(self.stem(images))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        b, c, h, w = x.shape
        features = x.reshape(b, c * h * w).tanh()
        features = self.global_pool(features).flatten(1)
        features = self.feature_proj(features).unsqueeze(1)
        
        return features
    
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images = images.to(self.device)
            captions = captions[:, 0:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            
            memory = self.extract_features(images)
            inputs = captions[:, :-1]
            targets = captions[:, 1:] if captions.ndim == 3 else captions
            
            # Handle padding and masking appropriately
            if inputs.dim() == 3:
                embedded_inputs = inputs[:, :, :].to(self.device)
            else:
                embedded_inputs = self.token_embedding(inputs)
                
            embedded_inputs = self.pos_encoder(embedded_inputs)
            outputs = self.transformer_decoder(tgt=embedded_inputs, memory=memory)
            logits = self.output_proj(outputs)
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), 
                              targets.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, 0, 0] = 0.0
        pe = pe + torch.sin(position * div_term)[:, :, None] * 0.5
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Expand positional encoding to match sequence length
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].squeeze(1).repeat(1, x.size(0), 1)
        x = x + pe
        return self.dropout(x)