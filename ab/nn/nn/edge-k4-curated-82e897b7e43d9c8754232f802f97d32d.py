import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms

def supported_hyperparameters():
    return {'batch', 'dropout', 'epoch', 'lr', 'momentum', 'transform'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.to(self.device)
        self.backbone = mobilenet_v3_small(weights='DEFAULT')
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(num_features, out_shape[0])
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def infer_dimensions_dynamically(self):
        dummy = torch.zeros(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
        with torch.no_grad():
            output = self(dummy)
        return output.shape

    def train_setup(self, prm):
        self._batch_size = int(prm['batch'])
        self._epochs = int(prm['epoch'])
        self._dropout = float(prm['dropout'])
        self._transform = prm['transform']
        self.to(self.device)
        learning_rate = float(prm['lr'])
        momentum = float(prm['momentum'])
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = (inputs.to(self.device), labels.to(self.device))
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()