import math
        import torch
        import torch.nn as nn
import torch.nn.functional as F
        from typing import Optional

        def calculate_kl(mu_q, sig_q, mu_p, sig_p):
            kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
            return kl

        class ModuleWrapper(nn.Module):
            def __init__(self):
                super(ModuleWrapper, self).__init__()
            
            def set_flag(self, flag_name, value):
                setattr(self, flag_name, value)
                for m in self.children():
                    if hasattr(m, 'set_flag'):
                        m.set_flag(flag_name, value)
            
            def forward(self, x):
                for module in self.children():
                    x = module(x)
                kl = 0.0
                for module in self.modules():
                    if hasattr(module, 'kl_loss'):
                        kl = kl + module.kl_loss()
                return x, kl

        class FlattenLayer(ModuleWrapper):
            def __init__(self, num_features):
                super(FlattenLayer, self).__init__()
                self.num_features = num_features
            
            def forward(self, x):
                return x.view(-1, self.num_features)

        class CNN_Encoder(ModuleWrapper):
            def __init__(self, input_channels=3, hidden_dim=768):
                super(CNN_Encoder, self).__init__()
                self.input_channels = input_channels
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.avgpool(x)
                x = x.squeeze(3).squeeze(2)  # Collapse spatial dimensions
                return x.unsqueeze(1)

        class BBBLinear(ModuleWrapper):
            def __init__(self, in_features, out_features, bias=True, priors=None):
                super(BBBLinear, self).__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.use_bias = bias
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            def forward(self, x, sample=True):
                self.W_sigma = torch.log1p(torch.exp(self.W_rho))
                if self.use_bias:
                    self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                    bias_var = self.bias_sigma ** 2
                else:
                    self.bias_sigma = bias_var = None
        
                act_mu = F.linear(x, self.W_mu, self.bias_mu)
                act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
                act_std = torch.sqrt(act_var)
        
                if self.training or sample:
                    eps = torch.empty_like(act_std).normal_(0, 1)
                    return act_mu + act_std * eps
                else:
                    return act_mu

        class Attention(nn.Module):
            def __init__(self, embed_dim, num_heads=8, kq_same_dim=True, v_same_dim=True, qkv_bias=False):
                super(Attention, self).__init__()
                self.num_heads = num_heads
                head_dim = embed_dim // num_heads
        
                self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
                self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
                self.scale = 1.0 / math.sqrt(head_dim)
        
            def forward(self, x, proj_queries=None, memory_keys=None, memory_values=None, mask=None):
                batch_size, seq_len, embed_dim = x.size()
        
                keys = self.k_proj(x)
                if proj_queries is not None:
                    queries = self.q_proj(proj_queries)
                else:
                    queries = self.q_proj(x)
                values = self.v_proj(x)
                if memory_keys is not None:
                    keys = torch.cat([memory_keys, keys], dim=1)
                    values = torch.cat([memory_values, values], dim=1)
                    seq_len_combined = proj_queries.size(1) + seq_len
                    mask_extended = torch.cat([
                        (mask[:, None, None] if mask is not None else None),
                        torch.full((batch_size, 1, seq_len_combined), float('-inf'))
                    ], dim=-1).view(batch_size, seq_len_combined, seq_len_combined)
                else:
                    seq_len_combined = seq_len
        
                q = queries.view(batch_size, self.num_heads, seq_len_combined, -1).permute(0, 2, 1, 3)
                k = keys.view(batch_size, self.num_heads, seq_len_combined, -1).permute(0, 2, 1, 3)
                v = values.view(batch_size, self.num_heads, seq_len_combined, -1).permute(0, 2, 1, 3)
        
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    scale=self.scale,
                    mask=(mask_extended if mask is not None else None)
                )
        
                return attn_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        class Decoder_Layer(nn.Module):  # Note: the user's code uses ModuleWrapper, but the class Decoder_Layer is defined as nn.Module
            def __init__(self, d_model, d_ff, nheads):
                super(Decoder_Layer, self).__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nheads)
                self.cross_attn = Attention(d_model, nheads)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_ff, d_model)
                )
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, memory, mask_self=None):
                x_skip = x
                x_norm = self.norm1(x_skip + self.dropout(self.self_attn(x, key_padding_mask=mask_self)[0]))
                x = x_norm + x_skip
        
                x_skip = x
                x_norm = self.norm2(x_skip + self.dropout(self.cross_attn(x, None, memory, None, None)[0]))
                x = x_norm + x_skip
        
                return x

        class Transformer_Decoder(nn.Module):
            def __init__(self, input_vocab_size, d_model, n_layers, nheads):
                super(Transformer_Decoder, self).__init__()
                self.input_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
                self.positional_encoding = nn.Parameter(torch.zeros(500, d_model)))
                self.n_layers = n_layers
                self.d_model = d_model
                self.num_heads = nheads
            
            def forward(self, input_ids, memory, hidden_states=None):
                batch_size = input_ids.size(0)
        
                x = input_ids
                x = self.input_embedding(x)
                x = x + self.positional_encoding[:x.size(1), :]
        
                if hidden_states is not None:
                    mask_length = hidden_states.size(1)
                    mask_self = torch.triu(
                        torch.full((mask_length, mask_length), float('-inf'), device=x.device),
                        diagonal=1
                    ).bool()
                else:
                    mask_self = None
        
                x = self.dec_layers(x, memory, mask_self)
        
                logits = self.final_proj(x)
                return logits, x[:,-1]  # Logits shape [batch, seq, vocab], hidden_state shape [batch, 1, d_model]

        def supported_hyperparameters():
    return {'lr','momentum'}


        class Net(nn.Module):
            def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device):
                super(Net, self).__init__()
                self.device = device
            
                input_channels = int(in_shape[1]) if in_shape else 3
                output_classes = int(out_shape[0])
                hidden_dim_val = int(prm.get('hidden_dim', 768))
                num_layers_val = int(prm.get('num_layers', 6))
                num_heads_val = int(prm.get('num_heads', 8))
        
                self.encoder = CNN_Encoder(input_channels=input_channels, hidden_dim=hidden_dim_val)
                self.decoder = Transformer_Decoder(
                    input_vocab_size=output_classes,
                    d_model=hidden_dim_val,
                    n_layers=num_layers_val,
                    nheads=num_heads_val
                )
        
                self.criterion = nn.CrossEntropyLoss(ignore_index=0)
                self.optimizer = None
        
            def init_zero_hidden(self, batch: int, device: torch.device):
                return torch.empty(0, device=device), torch.empty(0, device=device)
        
            def train_setup(self, prm):
                self.to(self.device)
                self.criterion = self.criterion.to(self.device)
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(prm.get('lr', 1e-3)), weight_decay=1e-4)
        
            def learn(self, train_data):
                self.train()
                for images, captions in train_data:
                    images = images.to(self.device, dtype=torch.float32)
                    targets = captions.to(self.device)
                    memory = self.encoder(images)
                    input_ids = targets[:, :-1]
                    output_targets = targets[:, 1:].reshape(-1)
                    decoder_output, _ = self.decoder(input_ids, memory)
                    decoder_output_reshaped = decoder_output.reshape(-1, decoder_output.size(-1))
                    loss = self.criterion(decoder_output_reshaped, output_targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
                    self.optimizer.step()
                
                return loss, decoder_output_reshaped[:len(output_targets_reshaped)]