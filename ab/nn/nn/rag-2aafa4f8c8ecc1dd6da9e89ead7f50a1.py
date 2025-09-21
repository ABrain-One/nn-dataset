# Auto-generated single-file for DenseDilatedKnnGraph
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- mmpretrain.models.backbones.vig.DenseDilated ----
class DenseDilated(nn.Module):
    """Find dilated neighbor from neighbor list.

    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, use_stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.use_stochastic = use_stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.use_stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

# ---- mmpretrain.models.backbones.vig.xy_pairwise_distance ----
def xy_pairwise_distance(x, y):
    """Compute pairwise distance of a point cloud.

    Args:
        x: tensor (batch_size, num_points, num_dims)
        y: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)

# ---- mmpretrain.models.backbones.vig.xy_dense_knn_matrix ----
def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.

    Args:
        x: (batch_size, num_dims, num_points, 1)
        y: (batch_size, num_dims, num_points, 1)
        k: int
        relative_pos:Whether to use relative_pos
    Returns:
        nearest neighbors:
        (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(
            0, n_points, device=x.device).repeat(batch_size, k,
                                                 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

# ---- DenseDilatedKnnGraph (target) ----
class DenseDilatedKnnGraph(nn.Module):
    """Find the neighbors' indices based on dilated knn."""

    def __init__(self, k=9, dilation=1, use_stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.use_stochastic = use_stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, use_stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)

            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation,
                                             relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            y = x.clone()

            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation,
                                             relative_pos)
        return self._dilated(edge_index)

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
        
        self.dense_dilated_knn = DenseDilatedKnnGraph(k=9, dilation=2, use_stochastic=False, epsilon=0.0)

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
