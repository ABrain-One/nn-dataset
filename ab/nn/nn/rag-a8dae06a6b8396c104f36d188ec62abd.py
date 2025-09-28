# Auto-generated single-file for GPSENodeEncoder
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- GPSENodeEncoder (target) ----
class GPSENodeEncoder(torch.nn.Module):
    r"""A helper linear/MLP encoder that takes the :class:`GPSE` encodings
    (based on the `"Graph Positional and Structural Encoder"
    <https://arxiv.org/abs/2307.07107>`_ paper) precomputed as
    :obj:`batch.pestat_GPSE` in the input graphs, maps them to a desired
    dimension defined by :obj:`dim_pe_out` and appends them to node features.

    Let's say we have a graph dataset with 64 original node features, and we
    have generated GPSE encodings of dimension 32, i.e.
    :obj:`data.pestat_GPSE` = 32. Additionally, we want to use a GNN with an
    inner dimension of 128. To do so, we can map the 32-dimensional GPSE
    encodings to a higher dimension of 64, and then append them to the
    :obj:`x` attribute of the :class:`~torch_geometric.data.Data` objects to
    obtain a 128-dimensional node feature representation.
    :class:`~torch_geometric.nn.GPSENodeEncoder` handles both this mapping and
    concatenation to :obj:`x`, the outputs of which can be used as input to a
    GNN:

    .. code-block:: python

        encoder = GPSENodeEncoder(dim_emb=128, dim_pe_in=32, dim_pe_out=64,
                                  expand_x=False)
        gnn = GNN(...)

        for batch in loader:
            x = encoder(batch.x, batch.pestat_GPSE)
            batch = gnn(x, batch.edge_index)

    Args:
        dim_emb (int): Size of final node embedding.
        dim_pe_in (int): Original dimension of :obj:`batch.pestat_GPSE`.
        dim_pe_out (int): Desired dimension of :class:`GPSE` after the encoder.
        dim_in (int, optional): Original dimension of input node features,
            required only if :obj:`expand_x` is set to :obj:`True`.
            (default: :obj:`None`)
        expand_x (bool, optional): Expand node features :obj:`x` from
            :obj:`dim_in` to (:obj:`dim_emb` - :obj:`dim_pe_out`)
        norm_type (str, optional): Type of normalization to apply.
            (default: :obj:`batchnorm`)
        model_type (str, optional): Type of encoder, either :obj:`mlp` or
            :obj:`linear`. (default: :obj:`mlp`)
        n_layers (int, optional): Number of MLP layers if :obj:`model_type` is
            :obj:`mlp`. (default: :obj:`2`)
        dropout_be (float, optional): Dropout ratio of inputs to encoder, i.e.
            before encoding. (default: :obj:`0.5`)
        dropout_ae (float, optional): Dropout ratio of outputs, i.e. after
            encoding. (default: :obj:`0.2`)
    """
    def __init__(self, dim_emb: int, dim_pe_in: int, dim_pe_out: int,
                 dim_in: int = None, expand_x=False, norm_type='batchnorm',
                 model_type='mlp', n_layers=2, dropout_be=0.5, dropout_ae=0.2):
        super().__init__()

        assert dim_emb > dim_pe_out, ('Desired GPSE dimension (dim_pe_out) '
                                      'must be smaller than the final node '
                                      'embedding dimension (dim_emb).')

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe_out)
        self.expand_x = expand_x

        self.raw_norm = None
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(dim_pe_in)

        self.dropout_be = nn.Dropout(p=dropout_be)
        self.dropout_ae = nn.Dropout(p=dropout_ae)

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(torch.nn.Linear(dim_pe_in, dim_pe_out))
                layers.append(activation())
            else:
                layers.append(torch.nn.Linear(dim_pe_in, 2 * dim_pe_out))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(
                        torch.nn.Linear(2 * dim_pe_out, 2 * dim_pe_out))
                    layers.append(activation())
                layers.append(torch.nn.Linear(2 * dim_pe_out, dim_pe_out))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(dim_pe_in, dim_pe_out)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, x, pos_enc):
        pos_enc = self.dropout_be(pos_enc)
        pos_enc = self.raw_norm(pos_enc) if self.raw_norm else pos_enc
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe
        pos_enc = self.dropout_ae(pos_enc)

        # Expand node features if needed
        h = self.linear_x(x) if self.expand_x else x

        # Concatenate final PEs to input embedding
        return torch.cat((h, pos_enc), 1)

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
        self.gpse_encoder = GPSENodeEncoder(dim_emb=64, dim_pe_in=16, dim_pe_out=32, dim_in=32, expand_x=True)
        self.classifier = nn.Linear(64, self.num_classes)

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
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        batch_size = x.size(0)
        pos_enc = torch.randn(batch_size, 16, device=x.device)
        x = self.gpse_encoder(x, pos_enc)
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
