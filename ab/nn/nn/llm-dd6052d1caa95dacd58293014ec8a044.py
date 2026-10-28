import torch
import torch.nn as nn
import numpy as np

def supported_hyperparameters():
    return {'lr', 'dropout', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.use_amp = prm.get('use_amp', False)
        if len(in_shape) == 4:
            self._input_spec = (in_shape[1], in_shape[2], in_shape[3])
        else:
            self._input_spec = (in_shape[0], in_shape[1], in_shape[2])
        in_channels = self._input_spec[0]
        self.features = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=prm.get('dropout', 0.1)), nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(p=prm.get('dropout', 0.1)), nn.Linear(512, out_shape[0]))
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _norm4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._norm4d(x)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.use_amp = prm.get('use_amp', False)

    def learn(self, train_data):
        self.train()
        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            for inputs, labels in train_data:
                inputs, labels = (inputs.to(self.device), labels.to(self.device))
                with torch.cuda.amp.autocast():
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
        else:
            for inputs, labels in train_data:
                inputs, labels = (inputs.to(self.device), labels.to(self.device))
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                self.optimizer.step()
        del train_data