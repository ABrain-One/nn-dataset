class EncoderBlock(nn.Module):
            def __init__(
                    self,
                    num_heads: int,
                    hidden_dim: int,
                    mlp_dim: int,
                    dropout: float,
                    attention_dropout: float,
                    norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            ):
                super().__init__()
                self.num_heads = num_heads

                self.ln_1 = norm_layer(hidden_dim)
                self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
                self.dropout = nn.Dropout(dropout)

                self.ln_2 = norm_layer(hidden_dim)
                self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

            def forward(self, input: torch.Tensor):
                torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
                x = self.ln_1(input)
                x, _ = self.self_attention(x, x, x, need_weights=False)
                x = self.dropout(x)
                x = x + input

                y = self.ln_2(x)
                y = self.mlp(y)
                return x + y

def supported_hyperparameters():
    return {'lr','momentum'}
