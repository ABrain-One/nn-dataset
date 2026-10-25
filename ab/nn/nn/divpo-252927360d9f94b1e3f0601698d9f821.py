import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[1], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=prm.get('dropout', 0.2))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(64 * 6 * 6, out_shape[0])
        self.dropout = nn.Dropout(p=prm.get('dropout', 0.5))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.stem(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9),
            weight_decay=prm.get('weight_decay', 0.0001)
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

                                   

    def get_complexity(self, input_shape):
        # Calculate the complexity of the model
        # The complexity is based on the number of parameters and the operations in the model
        # For simplicity, we use the number of parameters as the complexity
        complexity = self.get_num_params()
        return complexity

    def get_num_params(self):
        # Calculate the number of parameters in the model
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_params

    def get_memory_usage(self, input_shape):
        # Estimate the memory usage of the model
        # The memory usage is based on the input shape and the operations in the model
        # For simplicity, we use the input size times a factor as the memory usage
        memory_usage = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4 / (1024 ** 2)
        return memory_usage

    def get_latency(self, input_shape):
        # Estimate the latency of the model
        # The latency is based on the complexity and the operations in the model
        # For simplicity, we use the complexity times a factor as the latency
        latency = self.get_complexity(input_shape) * 0.001
        return latency

    def get_efficiency(self, input_shape):
        # Calculate the efficiency of the model
        # The efficiency is the ratio of the memory usage to the latency
        efficiency = self.get_memory_usage(input_shape) / self.get_latency(input_shape)
        return efficiency

    def get_complexity_per_param(self, input_shape):
        # Calculate the complexity per parameter
        # The complexity per parameter is the complexity divided by the number of parameters
        complexity_per_param = self.get_complexity(input_shape) / self.get_num_params()
        return complexity_per_param

    def get_efficiency_per_param(self, input_shape):
        # Calculate the efficiency per parameter
        # The efficiency per parameter is the efficiency divided by the complexity per parameter
        efficiency_per_param = self.get_efficiency(input_shape) / self.get_complexity_per_param(input_shape)
        return efficiency_per_param