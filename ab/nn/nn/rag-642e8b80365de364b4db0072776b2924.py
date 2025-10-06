# Auto-generated single-file for KGEModel
# Dependencies are emitted in topological order (utilities first).

# Standard library and external imports
import torch
from torch import Tensor
from typing import List, Tuple
from typing import List
from typing import Tuple

# ---- original imports from contributing modules ----
from tqdm import tqdm
from torch.nn import Embedding

# ---- torch_geometric.nn.kge.loader.KGTripletLoader ----
class KGTripletLoader(torch.utils.data.DataLoader):
    def __init__(self, head_index: Tensor, rel_type: Tensor,
                 tail_index: Tensor, **kwargs):
        self.head_index = head_index
        self.rel_type = rel_type
        self.tail_index = tail_index

        super().__init__(range(head_index.numel()), collate_fn=self.sample,
                         **kwargs)

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.head_index.device)

        head_index = self.head_index[index]
        rel_type = self.rel_type[index]
        tail_index = self.tail_index[index]

        return head_index, rel_type, tail_index

# ---- KGEModel (target) ----
class KGEModel(torch.nn.Module):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the score for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the loss value for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loader(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ) -> Tensor:
        r"""Returns a mini-batch loader that samples a subset of triplets.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            **kwargs (optional): Additional arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last`
                or :obj:`num_workers`.
        """
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    def test(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float, float]:
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
        """
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            rank = int((torch.cat(scores).argsort(
                descending=True) == t).nonzero().view(-1))
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k

    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')

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
        self.kge_model = KGEModel(num_nodes=100, num_relations=10, hidden_channels=32)
        self.classifier = torch.nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            torch.nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=False),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        head_index = torch.randint(0, 100, (x.size(0),), device=x.device)
        rel_type = torch.randint(0, 10, (x.size(0),), device=x.device)
        tail_index = torch.randint(0, 100, (x.size(0),), device=x.device)
        
        kge_features = self.kge_model.node_emb(head_index) + self.kge_model.rel_emb(rel_type)
        x_combined = x + kge_features
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
            self.optimizer.step()
