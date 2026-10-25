import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(32 * 6 * 6, out_shape[0])

        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_memory_breakdown(self):
        return {"trainable": self.count_params(), "fixed": 0, "total": 500000}

    def get_flops(self, input_shape):
        flops = 0
        x = torch.rand(input_shape, device=self.device)

        # Conv1 FLOPS
        flops += x.numel() * self.conv1.weight.numel()

        # ReLu FLOPS
        flops += x.numel()

        # Maxpool FLOPS
        flops += x.numel()

        # Avgpool FLOPS
        flops += x.numel()

        # Dropout FLOPS
        flops += x.numel()

        # Classifier FLOPS
        flops += x.numel() * self.classifier.weight.numel()

        return flops

    def get_latency(self, input_shape):
        x = torch.rand(input_shape, device=self.device)
        with torch.no_grad():
            # Conv1 Latency
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record_async()
            _ = self.conv1(x)
            end.record_async()
            end.synchronize()
            conv1_latency = start.elapsed_time(end)

            # ReLu Latency
            start.record_async()
            _ = self.relu(x)
            end.record_async()
            end.synchronize()
            relu_latency = start.elapsed_time(end)

            # Maxpool Latency
            start.record_async()
            _ = self.maxpool(x)
            end.record_async()
            end.synchronize()
            maxpool_latency = start.elapsed_time(end)

            # Avgpool Latency
            start.record_async()
            _ = self.avgpool(x)
            end.record_async()
            end.synchronize()
            avgpool_latency = start.elapsed_time(end)

            # Dropout Latency
            start.record_async()
            _ = self.dropout(x)
            end.record_async()
            end.synchronize()
            dropout_latency = start.elapsed_time(end)

            # Classifier Latency
            start.record_async()
            _ = self.classifier(x.view(x.size(0), -1))
            end.record_async()
            end.synchronize()
            classifier_latency = start.elapsed_time(end)

        total_latency = conv1_latency + relu_latency + maxpool_latency + avgpool_latency + dropout_latency + classifier_latency
        return total_latency