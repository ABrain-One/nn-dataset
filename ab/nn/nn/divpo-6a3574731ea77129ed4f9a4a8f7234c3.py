import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class DumbbellBlock(nn.Module):
    """A simple block with two paths: one main path and a shortcut path."""
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return nn.functional.relu(out)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.seen_colors = nn.Conv2d(self.in_channels, 16, kernel_size=1)
        self.dumbbell = DumbbellBlock(16, 32, prm['dropout'])
        self.getting_warmed_up = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.now_i_see_colors = nn.AvgPool2d((5, 5))
        self.my_classifier_is_small = nn.Linear(32, self.num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.seen_colors(x))
        x = self.dumbbell(x)
        x = self.getting_warmed_up(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
        x = self.now_i_see_colors(x)
        x = x.squeeze((2, 3))
        return self.my_classifier_is_small(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()