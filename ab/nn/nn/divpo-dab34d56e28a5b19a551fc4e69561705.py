import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = 32
        self.image_size = in_shape[2]
        self.classes = out_shape[0]
        self.first_conv = nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1)
        self.middle_blocks = 3
        self.after_block_channels = 32
        self.average_pool_size = 32
        self.classifier : nn.Module | None = None
        self._make_classifier()

    def _make_classifier(self):
        self.classifier = nn.Linear(self.after_block_channels, self.classes)

    def forward(self, x):
        x = self.first_conv(x)
        return self.classifier(x.sum(dim=(2, 3)))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()