# Auto-generated single-file for BoneLoss
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

# ---- BoneLoss (target) ----
class BoneLoss(nn.Module):
    """Bone length loss.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        use_target_weight (bool): Option to use weighted bone loss.
            Different bone types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 joint_parents,
                 use_target_weight: bool = False,
                 loss_weight: float = 1.,
                 loss_name: str = 'loss_bone'):
        super().__init__()
        self.joint_parents = joint_parents
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)

        self._loss_name = loss_name

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K-1]):
                Weights across different bone types.
        """
        output_bone = torch.norm(
            output - output[:, self.joint_parents, :],
            dim=-1)[:, self.non_root_indices]
        target_bone = torch.norm(
            target - target[:, self.joint_parents, :],
            dim=-1)[:, self.non_root_indices]
        if self.use_target_weight:
            assert target_weight is not None
            target_weight = target_weight[:, self.non_root_indices]
            loss = torch.mean(
                torch.abs((output_bone * target_weight).mean(dim=0) -
                          (target_bone * target_weight).mean(dim=0)))
        else:
            loss = torch.mean(
                torch.abs(output_bone.mean(dim=0) - target_bone.mean(dim=0)))

        return loss * self.loss_weight

    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def supported_hyperparameters():
    return {'lr','momentum'}


class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.features = self.build_features()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        layers += [
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]

        self._last_channels = 32
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
