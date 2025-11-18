import torch
import torch.nn as nn
import torch.nn.functional as F
def supported_hyperparameters():
    return {'lr','momentum'}



class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=768, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        
        # Base blocks
        block_expansion = 4
        
        # Standard block composition
        def Block(channels_in, channels_out, downsampling=False):
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU()
            )
        
        # Downsampling block for reducing dimensions while maintaining field-of-view
        def DownBlock(channels_in, channels_out):
            return nn.Sequential(
                Block(channels_in, channels_out//block_expansion, downsampling=True),
                nn.Conv2d(channels_out//block_expansion, channels_out, kernel_size=2, stride=2)
            )
        
        # Build encoder body (similar to ResNet but configurable)
        self.body = nn.Sequential()
        prev_channels = in_channels
        
        # Progressive doubling configuration
        config = [16, 32, 64, 128, 256]
        for i in range(len(config)-1):
            ch_in = prev_channels
            ch_out = config[i] * 4**i
            self.body.add_module(f'layer_{i}', DownBlock(ch_in, ch_out))
            prev_channels = ch_out
            
        # Final layer handling
        self.body.add_module('final_layer', Block(prev_channels, prev_channels * 2))
        
        # Classifier-free guidance conditioning term
        self.cond_term = nn.Sequential(
            nn.Linear(4, 256),  # Small conditioning network
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Attention refinement for contextual guidance
        self.attention_refine = nn.Sequential(
            nn.Linear(output_dim, output_dim//2),
            nn.ReLU(),
            nn.Linear(output_dim//2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, images, cond_scale=None):
        # Apply body transformations
        feat = self.body(images)
        
        # Shape adjustment: [B, C, H, W] → [B, C, 7, 7]
        B, C, H, W = feat.shape
        feat = F.avg_pool2d(feat, 7)  # Reduce spatial dimensions
        feat_flat = feat.view(B, C*H*W)  # Collapse to feature vector
        
        # Project to output dimension
        proj = nn.Linear(C*H*W, self.output_dim)(feat_flat)
        
        # Optional conditioning for CFG (Classifier-Free Guidance)
        if cond_scale is not None:
            cond = self.cond_term(cond_scale)
            refine_mask = self.attention_refine(cond_scale)
            proj = proj * refine_mask + cond  # Weighted fusion
        
        return proj.unsqueeze(1)  # [B, 1, output_dim] for decoder consistency


class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape
        
        # Encoder customization
        self.encoder_depth_mult = prm.get('depth_multiplier', 1)
        encoder_dim = int(768 * (1 + self.encoder_depth_mult))  # Dynamic dimensional scaling
        
        # Decoder customization
        decoder_dim = encoder_dim
        self.attention_heads = min(8, decoder_dim // 64)  # Head count proportional to dim
        
        # Build encoder
        self.encoder = CNNEncoder(
            in_channels=in_shape[1], 
            output_dim=encoder_dim,
            device=device
        ).apply(lambda m: setattr(m, 'bias', False))
        
        # Build decoder backbone
        def CustomAttentionDecoderLayer(channels):
            layer = nn.TransformerDecoderLayer(
                d_model=channels,
                nhead=min(channels // 64, 8),
                dim_feedforward=max(1024, channels*4),
                batch_first=True
            )
            return layer
        
        # Modified decoder with conditional attention
        decoder_layers = [
            CustomAttentionDecoderLayer(encoder_dim * (1+i)) 
            for i in range(self.encoder_depth_mult)
        ]
        
        # Set up transformer decoder architecture
        self.transformer_decoder = nn.TransformerDecoder(
            nn.ModuleList(decoder_layers),
            norm=nn.LayerNorm(decoder_dim)
        )
        
        # Projection layers
        self.embeddings = nn.Linear(decoder_dim, decoder_dim)  # Embedding transformation
        self.pos_encoder = nn.Parameter(torch.zeros(1, decoder_dim))  # Learnable positional encodings
        self.classifier = nn.Linear(decoder_dim, out_shape)
        
    def forward(self, images, captions=None, hidden_state=None):
        memory = self.encoder(images)
        
        if captions is not None:
            embedded = self.embeddings(captions) + self.pos_encoder
            seq_len = captions.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=captures.device), diagonal=1).bool()
            mask = mask.logical_not()
            tgt_mask = ~mask if mask is not None else None
            
            # Incrementally decode using transformer layers
            dec_context = embedded.clone()
            hidden_state_input = embedded
            
            output = embedded
            for layer in self.transformer_decoder.layers:
                output = layer(output, memory, tgt_mask=tgt_mask)
            
            # Final layer projects to vocabulary space
            logits = self.classifier(output)
            
            # Adjust hidden_state according to transformer architecture conventions
            # Using a simplified representation capturing the cumulative effect
            if hidden_state is None:
                hidden_state = torch.zeros(dec_context.shape, device=self.device)
            else:
                hidden_state = dec_context.clone()
                
            return logits, hidden_state
        else:
            # Default initialization during inference
            pass

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999))

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            logits = None
            if hasattr(self, 'forward'):
                out = self.forward(images, captions)
                logits = out[0] if isinstance(out, tuple) else out
            if logits is None:
                continue
            tgt = (captions[:,0,:] if captions.ndim==3 else captions)[:,1:]
            loss = self.criteria[0](logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

            return None, torch.zeros(self.embeddings.weight.shape, device=self.device)