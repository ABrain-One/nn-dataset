# Auto-generated single-file for GNNLayer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from typing import List
from typing import Tuple

# ---- original imports from contributing modules ----
from torch import Tensor, nn

# ---- GNNLayer (target) ----
class GNNLayer(nn.Module):
    """GNN layer for SDMGR.

    Args:
        node_dim (int): Dimension of node embedding. Defaults to 256.
        edge_dim (int): Dimension of edge embedding. Defaults to 256.
    """

    def __init__(self, node_dim: int = 256, edge_dim: int = 256) -> None:
        super().__init__()
        self.in_fc = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.coef_fc = nn.Linear(node_dim, 1)
        self.out_fc = nn.Linear(node_dim, node_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes: Tensor, edges: Tensor,
                nums: List[int]) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            nodes (Tensor): Concatenated node embeddings.
            edges (Tensor): Concatenated edge embeddings.
            nums (List[int]): List of number of nodes in each batch.

        Returns:
            tuple(Tensor, Tensor):

            - nodes (Tensor): New node embeddings.
            - edges (Tensor): New edge embeddings.
        """
        start, cat_nodes = 0, []
        for num in nums:
            sample_nodes = nodes[start:start + num]
            cat_nodes.append(
                torch.cat([
                    sample_nodes.unsqueeze(1).expand(-1, num, -1),
                    sample_nodes.unsqueeze(0).expand(num, -1, -1)
                ], -1).view(num**2, -1))
            start += num
        cat_nodes = torch.cat([torch.cat(cat_nodes), edges], -1)
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        coefs = self.coef_fc(cat_nodes)

        start, residuals = 0, []
        for num in nums:
            residual = F.softmax(
                -torch.eye(num).to(coefs.device).unsqueeze(-1) * 1e9 +
                coefs[start:start + num**2].view(num, num, -1), 1)
            residuals.append(
                (residual *
                 cat_nodes[start:start + num**2].view(num, num, -1)).sum(1))
            start += num**2

        nodes += self.relu(self.out_fc(torch.cat(residuals)))
        return nodes, cat_nodes

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
        self.gnn_layer = GNNLayer(node_dim=32, edge_dim=32)
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
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        batch_size = x.size(0)
        nodes = x
        edges = torch.zeros(batch_size, 32, device=x.device)
        nums = [1] * batch_size
        
        nodes, _ = self.gnn_layer(nodes, edges, nums)
        return self.classifier(nodes)

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
