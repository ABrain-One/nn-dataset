# Auto-generated single-file for TwoWayAttentionBlock
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from torch import Tensor
import math

# ---- ultralytics.models.sam.modules.transformer.Attention ----
class Attention(nn.Module):
    """
    An attention layer with downscaling capability for embedding size after projection.

    This class implements a multi-head attention mechanism with the option to downsample the internal
    dimension of queries, keys, and values.

    Attributes:
        embedding_dim (int): Dimensionality of input embeddings.
        kv_in_dim (int): Dimensionality of key and value inputs.
        internal_dim (int): Internal dimension after downsampling.
        num_heads (int): Number of attention heads.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.

    Methods:
        _separate_heads: Separate input tensor into attention heads.
        _recombine_heads: Recombine separated attention heads.
        forward: Compute attention output for given query, key, and value tensors.

    Examples:
        >>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
        >>> q = torch.randn(1, 100, 256)
        >>> k = v = torch.randn(1, 50, 256)
        >>> output = attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int = None,
    ) -> None:
        """
        Initialize the Attention module with specified dimensions and settings.

        Args:
            embedding_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            downsample_rate (int, optional): Factor by which internal dimensions are downsampled.
            kv_in_dim (int | None, optional): Dimensionality of key and value inputs. If None, uses embedding_dim.

        Raises:
            AssertionError: If num_heads does not evenly divide the internal dim (embedding_dim / downsample_rate).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Separate the input tensor into the specified number of attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """Recombine separated attention heads into a single tensor."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to query, key, and value tensors with optional downsampling.

        Args:
            q (torch.Tensor): Query tensor with shape (B, N_q, embedding_dim).
            k (torch.Tensor): Key tensor with shape (B, N_k, embedding_dim).
            v (torch.Tensor): Value tensor with shape (B, N_k, embedding_dim).

        Returns:
            (torch.Tensor): Output tensor after attention with shape (B, N_q, embedding_dim).
        """
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)

# ---- ultralytics.nn.modules.transformer.MLPBlock ----
class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """
        Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))

# ---- TwoWayAttentionBlock (target) ----
class TwoWayAttentionBlock(nn.Module):
    """
    A two-way attention block for simultaneous attention to image and query points.

    This class implements a specialized transformer block with four main layers: self-attention on sparse inputs,
    cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of dense
    inputs to sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after token-to-image attention.
        mlp (MLPBlock): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after MLP block.
        norm4 (nn.LayerNorm): Layer normalization after image-to-token attention.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip positional encoding in the first layer.

    Methods:
        forward: Apply self-attention and cross-attention to queries and keys.

    Examples:
        >>> embedding_dim, num_heads = 256, 8
        >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
        >>> queries = torch.randn(1, 100, embedding_dim)
        >>> keys = torch.randn(1, 1000, embedding_dim)
        >>> query_pe = torch.randn(1, 100, embedding_dim)
        >>> key_pe = torch.randn(1, 1000, embedding_dim)
        >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Initialize a TwoWayAttentionBlock for simultaneous attention to image and query points.

        This block implements a specialized transformer layer with four main components: self-attention on sparse
        inputs, cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention
        of dense inputs to sparse inputs.

        Args:
            embedding_dim (int): Channel dimension of the embeddings.
            num_heads (int): Number of attention heads in the attention layers.
            mlp_dim (int, optional): Hidden dimension of the MLP block.
            activation (Type[nn.Module], optional): Activation function for the MLP block.
            attention_downsample_rate (int, optional): Downsampling rate for the attention mechanism.
            skip_first_layer_pe (bool, optional): Whether to skip positional encoding in the first layer.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply two-way attention to process query and key embeddings in a transformer block.

        Args:
            queries (torch.Tensor): Query embeddings with shape (B, N_queries, embedding_dim).
            keys (torch.Tensor): Key embeddings with shape (B, N_keys, embedding_dim).
            query_pe (torch.Tensor): Positional encodings for queries with same shape as queries.
            key_pe (torch.Tensor): Positional encodings for keys with same shape as keys.

        Returns:
            queries (torch.Tensor): Processed query embeddings with shape (B, N_queries, embedding_dim).
            keys (torch.Tensor): Processed key embeddings with shape (B, N_keys, embedding_dim).
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        
        self.two_way_attention = TwoWayAttentionBlock(
            embedding_dim=64,
            num_heads=4,
            mlp_dim=128,
            activation=nn.ReLU,
            attention_downsample_rate=2,
            skip_first_layer_pe=False
        )
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)) 
        B, C, H, W = x.shape
        queries = x.view(B, C, H * W).permute(0, 2, 1) 
        keys = queries.clone() 
        
        query_pe = torch.zeros_like(queries)
        key_pe = torch.zeros_like(keys)
        
        queries, keys = self.two_way_attention(queries, keys, query_pe, key_pe)
        
        x = queries.mean(dim=1)  
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
