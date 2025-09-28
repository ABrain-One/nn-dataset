# Auto-generated single-file for KDLoss
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

# ---- KDLoss (target) ----
class KDLoss(nn.Module):
    """PyTorch version of logit-based distillation from DWPose Modified from
    the official implementation.

    <https://github.com/IDEA-Research/DWPose>
    Args:
        weight (float, optional): Weight of dis_loss. Defaults to 1.0
    """

    def __init__(
        self,
        name,
        use_this,
        weight=1.0,
    ):
        super(KDLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.weight = weight

    def forward(self, pred, pred_t, beta, target_weight):
        ls_x, ls_y = pred
        lt_x, lt_y = pred_t

        lt_x = lt_x.detach()
        lt_y = lt_y.detach()

        num_joints = ls_x.size(1)
        loss = 0

        loss += (self.loss(ls_x, lt_x, beta, target_weight))
        loss += (self.loss(ls_y, lt_y, beta, target_weight))

        return loss / num_joints

    def loss(self, logit_s, logit_t, beta, weight):

        N = logit_s.shape[0]

        if len(logit_s.shape) == 3:
            K = logit_s.shape[1]
            logit_s = logit_s.reshape(N * K, -1)
            logit_t = logit_t.reshape(N * K, -1)

        # N*W(H)
        s_i = self.log_softmax(logit_s * beta)
        t_i = F.softmax(logit_t * beta, dim=1)

        # kd
        loss_all = torch.sum(self.kl_loss(s_i, t_i), dim=1)
        loss_all = loss_all.reshape(N, K).sum(dim=1).mean()
        loss_all = self.weight * loss_all

        return loss_all

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
        self.kd_loss = KDLoss(name='kd', use_this=True, weight=1.0)
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
        
        pred_x = torch.randn(x.size(0), 17, 64, device=x.device)
        pred_y = torch.randn(x.size(0), 17, 64, device=x.device)
        pred_t_x = torch.randn(x.size(0), 17, 64, device=x.device)
        pred_t_y = torch.randn(x.size(0), 17, 64, device=x.device)
        beta = 1.0
        target_weight = torch.ones(x.size(0), 17, device=x.device)
        
        kd_loss = self.kd_loss((pred_x, pred_y), (pred_t_x, pred_t_y), beta, target_weight)
        x_combined = x + kd_loss.unsqueeze(-1)
        return self.classifier(x_combined)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
