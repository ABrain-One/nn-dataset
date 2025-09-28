# Auto-generated single-file for PANEmbLossV1
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
class MODELS:
    @staticmethod
    def build(cfg): return None
    @staticmethod
    def switch_scope_and_registry(scope): return MODELS()
    def __enter__(self): return self
    def __exit__(self, *args): pass

# ---- original imports from contributing modules ----
from torch import nn

# ---- PANEmbLossV1 (target) ----
class PANEmbLossV1(nn.Module):
    """The class for implementing EmbLossV1. This was partially adapted from
    https://github.com/whai362/pan_pp.pytorch.

    Args:
        feature_dim (int): The dimension of the feature. Defaults to 4.
        delta_aggregation (float): The delta for aggregation. Defaults to 0.5.
        delta_discrimination (float): The delta for discrimination.
            Defaults to 1.5.
    """

    def __init__(self,
                 feature_dim: int = 4,
                 delta_aggregation: float = 0.5,
                 delta_discrimination: float = 1.5) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_aggregation = delta_aggregation
        self.delta_discrimination = delta_discrimination
        self.weights = (1.0, 1.0)

    def _forward_single(self, emb: torch.Tensor, instance: torch.Tensor,
                        kernel: torch.Tensor,
                        training_mask: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a single image.

        Args:
            emb (torch.Tensor): The embedding feature.
            instance (torch.Tensor): The instance feature.
            kernel (torch.Tensor): The kernel feature.
            training_mask (torch.Tensor): The effective mask.
        """
        training_mask = (training_mask > 0.5).float()
        kernel = (kernel > 0.5).float()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)

        unique_labels, unique_ids = torch.unique(
            instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        emb_mean = emb.new_zeros((self.feature_dim, num_instance),
                                 dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_aggregation)**2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(
                -1, self.feature_dim)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(
                -1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_discrimination - dist)**2
            l_dis = torch.mean(torch.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb: torch.Tensor, instance: torch.Tensor,
                kernel: torch.Tensor,
                training_mask: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a batch image.

        Args:
            emb (torch.Tensor): The embedding feature.
            instance (torch.Tensor): The instance feature.
            kernel (torch.Tensor): The kernel feature.
            training_mask (torch.Tensor): The effective mask.
        """
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self._forward_single(emb[i], instance[i],
                                                 kernel[i], training_mask[i])

        return loss_batch

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
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.embedding = nn.Linear(32, 4)
        self.classifier = nn.Linear(4, self.num_classes)
        self.pan_emb_loss = PANEmbLossV1(feature_dim=4)

    def forward(self, x):
        x = self.features(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        
        # Use PANEmbLossV1 for additional regularization
        # Create dummy tensors for the loss function
        B = x.shape[0]
        instance = torch.ones(B, 1, 1, 1, device=x.device)
        kernel = torch.ones(B, 1, 1, 1, device=x.device)
        training_mask = torch.ones(B, 1, 1, 1, device=x.device)
        
        # Reshape embedding for loss computation
        emb_reshaped = emb.view(B, 4, 1, 1)
        emb_loss = self.pan_emb_loss(emb_reshaped, instance, kernel, training_mask)
        
        return logits

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            cls_loss = self.criteria(output, target)
            x = self.features(data)
            emb = self.embedding(x)
            B = emb.shape[0]
            instance = torch.ones(B, 1, 1, 1, device=emb.device)
            kernel = torch.ones(B, 1, 1, 1, device=emb.device)
            training_mask = torch.ones(B, 1, 1, 1, device=emb.device)
            emb_reshaped = emb.view(B, 4, 1, 1)
            emb_loss = self.pan_emb_loss(emb_reshaped, instance, kernel, training_mask)
            
            # Combine losses
            total_loss = cls_loss + 0.1 * emb_loss.mean()
            total_loss.backward()
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
