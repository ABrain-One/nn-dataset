# Auto-generated single-file for DeformableTransformerDecoder
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import copy

# ---- ultralytics.nn.modules.utils._get_clones ----
def _get_clones(module, n):
    """
    Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

# ---- ultralytics.nn.modules.utils.inverse_sigmoid ----
def inverse_sigmoid(x, eps=1e-5):
    """
    Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# ---- DeformableTransformerDecoder (target) ----
class DeformableTransformerDecoder(nn.Module):
    """
    Deformable Transformer Decoder based on PaddleDetection implementation.

    This class implements a complete deformable transformer decoder with multiple decoder layers and prediction
    heads for bounding box regression and classification.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Hidden dimension.
        eval_idx (int): Index of the layer to use during evaluation.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """
        Initialize the DeformableTransformerDecoder with the given parameters.

        Args:
            hidden_dim (int): Hidden dimension.
            decoder_layer (nn.Module): Decoder layer module.
            num_layers (int): Number of decoder layers.
            eval_idx (int): Index of the layer to use during evaluation.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # decoder embeddings
        refer_bbox: torch.Tensor,  # anchor
        feats: torch.Tensor,  # image features
        shapes: list,  # feature shapes
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        Perform the forward pass through the entire decoder.

        Args:
            embed (torch.Tensor): Decoder embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Image features.
            shapes (list): Feature shapes.
            bbox_head (nn.Module): Bounding box prediction head.
            score_head (nn.Module): Score prediction head.
            pos_mlp (nn.Module): Position MLP.
            attn_mask (torch.Tensor, optional): Attention mask.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            dec_bboxes (torch.Tensor): Decoded bounding boxes.
            dec_cls (torch.Tensor): Decoded classification scores.
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, self.num_classes)
        
        class SimpleDecoderLayer(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, hidden_dim)
                self.norm = nn.LayerNorm(hidden_dim)
            
            def forward(self, x, *args, **kwargs):
                return self.norm(x + self.linear(x))
        
        self.decoder_layer = SimpleDecoderLayer(32)
        self.deformable_decoder = DeformableTransformerDecoder(
            hidden_dim=32,
            decoder_layer=self.decoder_layer,
            num_layers=2,
            eval_idx=1
        )

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
