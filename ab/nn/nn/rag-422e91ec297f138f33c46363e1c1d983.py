# Auto-generated single-file for MPJPEVelocityJointLoss
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

# ---- MPJPEVelocityJointLoss (target) ----
class MPJPEVelocityJointLoss(nn.Module):
    """MPJPE (Mean Per Joint Position Error) loss.

    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
        lambda_scale (float): Factor of the N-MPJPE loss. Default: 0.5.
        lambda_3d_velocity (float): Factor of the velocity loss. Default: 20.0.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 lambda_scale=0.5,
                 lambda_3d_velocity=20.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.lambda_scale = lambda_scale
        self.lambda_3d_velocity = lambda_3d_velocity

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        norm_output = torch.mean(
            torch.sum(torch.square(output), dim=-1, keepdim=True),
            dim=-2,
            keepdim=True)
        norm_target = torch.mean(
            torch.sum(target * output, dim=-1, keepdim=True),
            dim=-2,
            keepdim=True)

        velocity_output = output[..., 1:, :, :] - output[..., :-1, :, :]
        velocity_target = target[..., 1:, :, :] - target[..., :-1, :, :]

        if self.use_target_weight:
            assert target_weight is not None
            mpjpe = torch.mean(
                torch.norm((output - target) * target_weight, dim=-1))

            nmpjpe = torch.mean(
                torch.norm(
                    (norm_target / norm_output * output - target) *
                    target_weight,
                    dim=-1))

            loss_3d_velocity = torch.mean(
                torch.norm(
                    (velocity_output - velocity_target) * target_weight,
                    dim=-1))
        else:
            mpjpe = torch.mean(torch.norm(output - target, dim=-1))

            nmpjpe = torch.mean(
                torch.norm(
                    norm_target / norm_output * output - target, dim=-1))

            loss_3d_velocity = torch.mean(
                torch.norm(velocity_output - velocity_target, dim=-1))

        loss = mpjpe + nmpjpe * self.lambda_scale + \
            loss_3d_velocity * self.lambda_3d_velocity

        return loss * self.loss_weight

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
        self.mpjpe_loss = MPJPEVelocityJointLoss(use_target_weight=False, loss_weight=1.0)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
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
            output = self.forward(data)
            loss = self.criteria(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()