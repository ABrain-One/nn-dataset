import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'dropout', 'momentum'}

class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(nn.Conv2d(in_shape[1], 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        dropout_prob = prm.get('dropout', 0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_prob), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(p=dropout_prob), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, out_shape[0]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = torch.cuda.amp.GradScaler(enabled=prm.get('use_amp', False))

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for inputs, labels in train_data:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
            if not torch.isfinite(loss):
                continue
            batch_size = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((outputs.detach().argmax(dim=1) == labels).sum().item())
            total_examples += batch_size
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            scaler.step(self.optimizer)
            scaler.update()
        if total_examples == 0:
            return (0.0, None)
        train_accuracy = total_correct / total_examples
        train_loss = total_loss / total_examples
        return (train_accuracy, train_loss)