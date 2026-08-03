import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}

class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.LazyLinear(channel // reduction, bias=False), nn.ReLU(inplace=True), nn.LazyLinear(channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.to(self.device)
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.dropout_rate = prm['dropout']
        base_model = models.mobilenet_v3_small(weights='DEFAULT')
        base_model.features[0][0] = nn.LazyConv2d(16, kernel_size=3, stride=2, padding=1)
        base_model.classifier[0] = nn.LazyLinear(1280)
        self.features = nn.Sequential(base_model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p=self.dropout_rate), nn.LazyLinear(self.num_classes))
        try:
            import torch as _edge_torch
            self.to(device)
            _edge_was_training = self.training
            self.eval()
            with _edge_torch.no_grad():
                self(_edge_torch.zeros((2, *tuple(in_shape)[1:]), device=device))
            if _edge_was_training:
                self.train()
        except Exception:
            pass
        _edge_lazy_materialized = True

    def infer_dimensions_dynamically(self, in_shape):
        dummy = torch.zeros(1, in_shape[1], in_shape[2], in_shape[3]).to(self.device)
        dummy_out = self.features(dummy)
        return dummy_out.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def train_setup(self, prm: dict):
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