import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        image_size = in_shape[2]
        num_channels = in_shape[1]
        hidden_dim = 640
        num_heads = 8
        mlp_dim = 3072
        num_layers = 6
        attention_dropout = 0.1
        dropout = 0.1

        self.conv_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=image_size//16, stride=1)
        patch_size = image_size // 16
        self.patch_size = patch_size
        seq_length = (image_size // patch_size) ** 2

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=attention_dropout, batch_first=True),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True),
            num_layers=1
        )

        self.embedding = nn.Embedding(out_shape[0], hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_shape[0])

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss(ignore_index=0).to(self.device),)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=prm['lr'], betas=(prm.get('momentum', 0.9), 0.999)
        )


    def learn(self, train_data):
        pass

    def forward(self, x, captions=None, hidden_state=None, teacher_forcing=True):
        pass

def supported_hyperparameters():
    return {'lr', 'momentum'}