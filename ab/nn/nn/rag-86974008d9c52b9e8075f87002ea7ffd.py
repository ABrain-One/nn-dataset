# Auto-generated single-file for MultipleLossWrapper
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

# ---- MultipleLossWrapper (target) ----
class MultipleLossWrapper(nn.Module):
    """A wrapper to collect multiple loss functions together and return a list
    of losses in the same order.

    Args:
        losses (list): List of Loss Config
    """

    def __init__(self, losses: list):
        super().__init__()
        self.num_losses = len(losses)

        loss_modules = []
        for loss_cfg in losses:
            t_loss = MODELS.build(loss_cfg)
            loss_modules.append(t_loss)
        self.loss_modules = nn.ModuleList(loss_modules)

    def forward(self, input_list, target_list, keypoint_weights=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            input_list (List[Tensor]): List of inputs.
            target_list (List[Tensor]): List of targets.
            keypoint_weights (Tensor[N, K, D]):
                Weights across different joint types.
        """
        assert isinstance(input_list, list), ''
        assert isinstance(target_list, list), ''
        assert len(input_list) == len(target_list), ''

        losses = []
        for i in range(self.num_losses):
            input_i = input_list[i]
            target_i = target_list[i]

            loss_i = self.loss_modules[i](input_i, target_i, keypoint_weights)
            losses.append(loss_i)

        return losses

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
        self.classifier = nn.Linear(32, self.num_classes)
        self.multiple_loss_wrapper = MultipleLossWrapper([
            {'type': 'CrossEntropyLoss'}
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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
            self.optimizer.step()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)
