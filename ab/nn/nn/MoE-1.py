import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Expert(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
        return x

class Gate(nn.Module):
    def __init__(self, input_dim, n_experts):
        super(Gate, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        n_experts = 3
        self.experts = [Expert(in_shape[0], out_shape[0]) for _ in range(n_experts)]
        self.gate = Gate(in_shape[0], n_experts)

    def forward(self, x):

        gates = self.gate(x).unsqueeze(-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.matmul(expert_outputs, gates).squeeze(-1)
        return output
