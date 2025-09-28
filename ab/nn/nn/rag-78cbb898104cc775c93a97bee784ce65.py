# Auto-generated single-file for SelectSeq
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- SelectSeq (target) ----
class SelectSeq(nn.Module):
    def __init__(self, mode='index', index=0):
        super(SelectSeq, self).__init__()
        self.mode = mode
        self.index = index

    def forward(self, x):
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    def forward(self, x):
        # type: (Tuple[torch.Tensor]) -> (torch.Tensor)
        pass

    def forward(self, x) -> torch.Tensor:
        if self.mode == 'index':
            return x[self.index]
        else:
            return torch.cat(x, dim=1)

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
        self.select_seq = SelectSeq(mode='index', index=0)
        self.classifier = nn.Linear(64, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        x_list = [x_flat[:, i:i+1, :] for i in range(C)]
        selected = self.select_seq(x_list)
        x = selected.mean(dim=2) 
        x = x.squeeze(1) 
        x = x.unsqueeze(1).repeat(1, 64)  
        x = self.classifier(x)
        return x

    def train_setup(self, prm: dict):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

    def learn(self, data_roll):
        for data, target in data_roll:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
