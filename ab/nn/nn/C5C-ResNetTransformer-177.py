import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = int(out_shape[0])  # Only need vocab size from out_shape
        self.hidden_dim = 768  # Set hidden dimension ≥640
        
        # Encoder: Residual network inspired by classification models
        expand = lambda c: int(c * prm.get('depth_multiple', 1.0))  # Expansion helper
        
        # Stems
        self.stem_conv = nn.Conv2d(
            in_shape[1],
            expand(32),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.stem_bn = nn.BatchNorm2d(expand(32))
        self.stem_act = nn.ReLU(inplace=True)
        
        # Mobile inverted blocks
        stage_info = [
            # Stage 1
            {'input_c': expand(32), 'out_c': expand(16), 't': 3, 's': 2, 'exp': 16},
            # Stage 2
            {'input_c': expand(16), 'out_c': expand(48), 't': 4, 's': 2, 'exp': 144},
            {'input_c': expand(48), 'out_c': expand(96), 't': 4, 's': 1, 'exp': 240},
            # Stage 3
            {'input_c': expand(96), 'out_c': expand(192), 't': 4, 's': 1, 'exp': 384},
            {'input_c': expand(192), 'out_c': expand(384), 't': 6, 's': 2, 'exp': 576},  # Add more stages if needed
        ]
        
        # Create blocks
        layers = []
        for idx, info in enumerate(stage_info):
            rep_num = info['t']  # Number of repetitions
            curr_out = info['out_c']
            
            for i_rep in range(rep_num):
                # Expansion convolution
                exp_c = expand(info['exp'])
                
                # Depthwise separable convolution block (expanded dimension version)
                inv = i_rep == 0
                
                conv = nn.Sequential()
                conv.add_module('expand', nn.Conv2d(info['input_c'], exp_c, kernel_size=1, padding=0, stride=1))
                conv.add_module('bn', nn.BatchNorm2d(exp_c))
                conv.add_module('act', nn.ReLU(inplace=True))
                
                dwconv = nn.Sequential()
                dwconv.add_module('dw', nn.Conv2d(
                    exp_c, 
                    exp_c, 
                    kernel_size=3, 
                    stride=info['s'],
                    padding=1,
                    groups=exp_c,
                    bias=False
                ))
                dwconv.add_module('dw_bn', nn.BatchNorm2d(exp_c))
                dwconv.add_module('dw_act', nn.ReLU(inplace=True))
                
                project = nn.Sequential()
                if inv or info['input_c'] != curr_out:
                    project.add_module('project', nn.Conv2d(exp_c, curr_out, kernel_size=1))
                else:
                    project.add_module('identity', nn.Identity())
                    
                layers.extend([
                    *layers,      # Keep previous layers
                    dwconv,       # Depthwise separable convolution
                    project       # Projecting convolution (or identity)
                ])
                
            # Update input channels for next stage
            info['input_c'] = curr_out
            
        self.enc_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head for encoder features (matching hidden_dim requirement)
        self.projector = nn.Conv2d(sum([i['out_c'] for i in stage_info]), self.hidden_dim, kernel_size=1)
        
    def init_zero_hidden(self, batch: int, device: torch.device):
        return torch.empty(0, device=device), torch.empty(0, device=device)
    
    def train_setup(self, prm: dict):
        self.to(self.device)
        # Use CrossEntropyLoss with proper ignoring index
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(self.device)
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=max(float(prm.get('lr', 1e-3)), 3e-4),
            weight_decay=min(float(prm.get('weight_decay', 0.01)), 0.01)
        )
        self.dropout_val = min(max(float(prm.get('dropout', 0.5)), 0.2), 0.8)
        
    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            # Move data to device
            images = images.to(self.device, dtype=torch.float32)
            caps = captions[:,0,:].long().to(self.device) if captions.ndim == 3 else captions.long().to(self.device)
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            
            # Forward pass
            memory = self.encode(images)
            logits, hidden_state = self.decode(inputs, memory)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
            
    def encode(self, images):
        """Extract features from image using shared encoder"""
        feat = self.stem_conv(F.relu(self.stem_bn(images)))
        feat = self.stem_act(feat)
        x = self.global_pool(feat)
        return self.projector(x)
        
    def decode(self, inputs, memory=None):
        """Decode using Transformer with cross attention on encoded features"""
        # Setup decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=int(self.hidden_dim*4),
            batch_first=True,
            dropout=self.dropout_val
        )
        decoder = nn.TransformerDecoder(dec_layer, num_layers=6)
        
        # Embedding layer
        embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Expand memory to sequence format [B, S, H] (currently [B, 1, H])
        expanded_mem = memory.unsqueeze(1)
        
        # Apply embedding and positional encoding
        embedded = embedding(inputs)
        
        # Generate masks and outputs
        mask_queries = torch.triu(
            torch.ones_like(embedded), 
            diagonal=1
        ).masked_fill(True, float('-inf'))
        
        # Combine memory and embedded features
        output = decoder(
            tgt=embedded,
            src=expanded_mem,
            memory_key_padding_mask=None,
            tgt_mask=mask_queries
        )
        
        # Apply final classification layer
        logits = F.linear(output, None, None)
        
        # Hidden state maintenance
        hidden_state = None
        
        return logits, hidden_state
        
    def forward(self, images: torch.Tensor, captions: Optional[torch.Tensor] = None, hidden_state=None):
        # Extract visual features
        img_feat = images.to(self.device)
        memory = self.encode(img_feat)
        
        # If captions are provided, process with teacher forcing
        if captions is not None:
            caps_inputs = captions[:, :-1]
            caps_targets = captions[:, 1:]
            logits, _ = self.decode(caps_inputs, memory)
            return logits, hidden_state
            
        # Else, prepare for decoding without captions
        # For demonstration purposes, we'll initialize hidden state properly
        # and set up necessary tensors for the actual API
        
        # This section ensures proper initialization according to the API
        bs = images.size(0)
        seq_len = 15  # Typical max caption length
        sos_idx = captions[:, 0] if captions is not None else torch.ones(bs, dtype=torch.long, device=self.device)
        
        # Create initial inputs with SOS tokens
        initial_input = sos_idx.to(self.device)
        initial_input = initial_input.repeat_interleave(seq_len - 1, dim=0).view(bs, seq_len - 1)
        
        logits, hidden_state = self.decode(initial_input, memory)
        return logits, hidden_state