# Auto-generated single-file for ImageEncoder
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- ImageEncoder (target) ----
class ImageEncoder(nn.Module):
    """
    Encode images using a trunk-neck architecture, producing multiscale features and positional encodings.

    This class combines a trunk network for feature extraction with a neck network for feature refinement
    and positional encoding generation. It can optionally discard the lowest resolution features.

    Attributes:
        trunk (nn.Module): The trunk network for initial feature extraction.
        neck (nn.Module): The neck network for feature refinement and positional encoding generation.
        scalp (int): Number of lowest resolution feature levels to discard.

    Methods:
        forward: Process the input image through the trunk and neck networks.

    Examples:
        >>> trunk = SomeTrunkNetwork()
        >>> neck = SomeNeckNetwork()
        >>> encoder = ImageEncoder(trunk, neck, scalp=1)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(image)
        >>> print(output.keys())
        dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """
        Initialize the ImageEncoder with trunk and neck networks for feature extraction and refinement.

        This encoder combines a trunk network for feature extraction with a neck network for feature refinement
        and positional encoding generation. It can optionally discard the lowest resolution features.

        Args:
            trunk (nn.Module): The trunk network for initial feature extraction.
            neck (nn.Module): The neck network for feature refinement and positional encoding generation.
            scalp (int): Number of lowest resolution feature levels to discard.

        Examples:
            >>> trunk = SomeTrunkNetwork()
            >>> neck = SomeNeckNetwork()
            >>> encoder = ImageEncoder(trunk, neck, scalp=1)
            >>> image = torch.randn(1, 3, 224, 224)
            >>> output = encoder(image)
            >>> print(output.keys())
            dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        """
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match."
        )

    def forward(self, sample: torch.Tensor):
        """Encode input through trunk and neck networks, returning multiscale features and positional encodings."""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        return {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }

class SimpleTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_list = [32, 32]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        return [x1, x2]

class SimpleNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_channel_list = [32, 32]
        self.conv = nn.Conv2d(32, 32, kernel_size=1, bias=False)
    
    def forward(self, features):
        processed_features = []
        pos_encodings = []
        for feat in features:
            processed = self.conv(feat)
            processed_features.append(processed)
            pos_encodings.append(torch.zeros_like(processed))
        return processed_features, pos_encodings

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
        self.trunk = SimpleTrunk()
        self.neck = SimpleNeck()
        self.image_encoder = ImageEncoder(trunk=self.trunk, neck=self.neck, scalp=0)
        self.classifier = nn.Linear(32, self.num_classes)

    def forward(self, x):
        output = self.image_encoder(x)
        vision_features = output['vision_features']
        x = nn.functional.adaptive_avg_pool2d(vision_features, (1, 1))
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
