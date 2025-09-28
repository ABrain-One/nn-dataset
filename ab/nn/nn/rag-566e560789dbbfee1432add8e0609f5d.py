# Auto-generated single-file for GCAModule
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- original imports from contributing modules ----

# ---- GCAModule (target) ----
class GCAModule(nn.Module):
    """GCAModule in MASTER.

    Args:
        in_channels (int): Channels of input tensor.
        ratio (float): Scale ratio of in_channels.
        n_head (int): Numbers of attention head.
        pooling_type (str): Spatial pooling type. Options are [``avg``,
            ``att``].
        scale_attn (bool): Whether to scale the attention map. Defaults to
            False.
        fusion_type (str): Fusion type of input and context. Options are
            [``channel_add``, ``channel_mul``, ``channel_concat``].
    """

    def __init__(self,
                 in_channels: int,
                 ratio: float,
                 n_head: int,
                 pooling_type: str = 'att',
                 scale_attn: bool = False,
                 fusion_type: str = 'channel_add',
                 **kwargs) -> None:
        super().__init__()

        assert pooling_type in ['avg', 'att']
        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']

        # in_channels must be divided by headers evenly
        assert in_channels % n_head == 0 and in_channels >= 8

        self.n_head = n_head
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.scale_attn = scale_attn
        self.single_header_inplanes = int(in_channels / n_head)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(
                self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            # for concat
            self.cat_conv = nn.Conv2d(
                2 * self.in_channels, self.in_channels, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial pooling function.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after spatial pooling.
        """
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.view(batch * self.n_head, self.single_header_inplanes,
                       height, width)
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            input_x = input_x.view(batch * self.n_head,
                                   self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.view(batch * self.n_head, 1,
                                             height * width)

            # scale variance
            if self.scale_attn and self.n_head > 1:
                context_mask = context_mask / \
                               torch.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] =
            # [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.view(batch,
                                   self.n_head * self.single_header_inplanes,
                                   1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output tensor after GCAModule.
        """
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = torch.cat([out,
                             channel_concat_term.expand(-1, -1, H, W)],
                            dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.in_channels, H, W])
            out = nn.functional.relu(out)

        return out

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
        self.gca_module = GCAModule(in_channels=32, ratio=0.25, n_head=8, pooling_type='att', fusion_type='channel_add')
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
        x = self.gca_module(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
