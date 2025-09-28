# Auto-generated single-file for FlamingoLayer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
from typing import Optional

# ---- original imports from contributing modules ----
from torch import nn

# ---- FlamingoLayer (target) ----
class FlamingoLayer(nn.Module):
    """Faminogo layers.

    Args:
        gated_cross_attn_layer (nn.Module): Gated cross attention layer.
        decoder_layer (nn.Module): Decoder layer.
    """

    def __init__(self, gated_cross_attn_layer: nn.Module,
                 decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    def condition_vis_x(self, vis_x):
        """Set condition vision features."""
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        """Set condition media locations."""
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous):
        """Set attend previous."""
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        """Forward function.

        Args:
            lang_x (torch.Tensor): language inputs.
            attention_mask (torch.Tensor, optional): text attention mask.
                Defaults to None.
            **decoder_layer_kwargs: Other decoder layer keyword arguments.
        """
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(
                lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError('vis_x must be conditioned before forward pass')

        if self.media_locations is None:
            raise ValueError(
                'media_locations must be conditioned before forward pass')

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
        )
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x

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
        
        class GatedCrossAttention(nn.Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
                self.gate = nn.Linear(embed_dim, embed_dim)
                
            def forward(self, lang_x, vis_x, media_locations=None, attend_previous=None):
                attn_out, _ = self.attention(lang_x, vis_x, vis_x)
                gate = torch.sigmoid(self.gate(lang_x))
                return lang_x + gate * attn_out
        
        class CustomDecoderLayer(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
                
            def forward(self, x, memory=None, attention_mask=None, **kwargs):
                if memory is None:
                    memory = x
                return self.decoder_layer(x, memory)
        
        self.gated_cross_attn = GatedCrossAttention(embed_dim=32, num_heads=4)
        self.decoder_layer = CustomDecoderLayer(d_model=32, nhead=4)
        self.flamingo_layer = FlamingoLayer(self.gated_cross_attn, self.decoder_layer)
        
        self.classifier = nn.Linear(32, self.num_classes)

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
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        # Reshape for transformer input
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Set up conditions for FlamingoLayer
        vis_x = x.clone()
        media_locations = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        self.flamingo_layer.condition_vis_x(vis_x)
        self.flamingo_layer.condition_media_locations(media_locations)
        self.flamingo_layer.condition_attend_previous(False)
        
        # Forward through FlamingoLayer
        x = self.flamingo_layer(x)
        x = x.squeeze(1)  # Remove sequence dimension
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
