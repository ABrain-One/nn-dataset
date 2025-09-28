# Auto-generated single-file for Embeddings
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import math

# ---- original imports from contributing modules ----

# ---- Embeddings (target) ----
class Embeddings(nn.Module):
    """Construct the word embeddings given vocab size and embed dim.

    Args:
        d_model (int): The embedding dimension.
        vocab (int): Vocablury size.
    """

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        """Forward the embeddings.

        Args:
            input (torch.Tensor): The input tensors.

        Returns:
            torch.Tensor: The embeddings.
        """
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)

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
        self.projection = nn.Linear(32, 1000)
        self.embeddings = Embeddings(d_model=32, vocab=1000)
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
        x = self.avgpool(x)
        x = x.flatten(1)
        
        x_proj = self.projection(x)
        indices = torch.argmax(x_proj, dim=1)
        x_embedded = self.embeddings(indices)
        
        return self.classifier(x_embedded)

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
