import torch
import torch.nn as nn
import torchvision.models as models

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)
        self.infer_dimensions_dynamically(in_shape, out_shape, prm)

    def infer_dimensions_dynamically(self, in_shape, out_shape, prm):
        dummy = torch.zeros(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
        self.backbone = models.mobilenet_v3_small(pretrained=False, weights='DEFAULT').features.to(self.device)
        dummy = self.backbone(dummy)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        dummy = self.global_avg_pool(dummy)
        flattened_size = dummy.view(1, -1).size(1)
        self.classifier = nn.Linear(flattened_size, out_shape[0]).to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._dropout = float(prm['dropout'])
        self._transform = prm['transform']
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()