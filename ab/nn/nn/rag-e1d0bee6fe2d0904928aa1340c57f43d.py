# Auto-generated single-file for EmbeddingBlock
# Dependencies are emitted in topological order (utilities first).
# UNRESOLVED DEPENDENCIES:
# Embedding, Linear, sqrt
# This block may not compile due to missing dependencies.

# Standard library and external imports
import torch
from torch import Tensor
from typing import Callable

# ---- original imports from contributing modules ----
from math import sqrt
from torch.nn import Embedding, Linear

# ---- EmbeddingBlock (target) ----
class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
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
        self.embedding_block = EmbeddingBlock(num_radial=16, hidden_channels=32, act=torch.nn.ReLU())

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        
        batch_size = x.shape[0]
        num_nodes = 4
        
        node_features = torch.randint(0, 95, (batch_size, num_nodes), device=x.device)
        rbf_features = torch.randn(batch_size, 16, device=x.device)
        edge_i = torch.randint(0, num_nodes, (batch_size,), device=x.device)
        edge_j = torch.randint(0, num_nodes, (batch_size,), device=x.device)
        
        x_emb = self.embedding_block.emb(node_features)
        rbf_emb = self.embedding_block.act(self.embedding_block.lin_rbf(rbf_features))
        
        x_i = x_emb[torch.arange(batch_size), edge_i]
        x_j = x_emb[torch.arange(batch_size), edge_j]
        
        x_embedded = self.embedding_block.act(self.embedding_block.lin(torch.cat([x_i, x_j, rbf_emb], dim=-1)))
        
        x_combined = x + 0.1 * x_embedded
        
        return self.classifier(x_combined)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = torch.nn.CrossEntropyLoss().to(self.device)
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
