import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_shape[1], 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, out_shape[0])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm.get('lr', 0.01),
            momentum=prm.get('momentum', 0.9)
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
            self.optimizer.step()

# The following code is used for the construction of the network and should not be modified
# (this allows the code to be run as-is to build the network)
# Constructing the network with `in_shape` and `prm` (which includes `prm['device']`) should
# set up the network for training on C x H x W images with the given hyperparameters.
# The network should be built according to the above specification, with as few
# layers and as few parameters as possible to keep the volume below 500000.
# No pretrained weights may be used.

# For example, given `in_shape` (1, 32, 32) and `prm` containing `{'lr': 0.01, 'momentum': 0.9, 'device': 'cuda'}`,
# your job is to produce a new (or overridden) `Net` class with a `__call__` method that defines the network,
# and an `train_setup` method that defines the loss function, optimizer, and device.
# The network itself must not exceed 500000 parameters, and must use standard PyTorch layers only.
# This includes: nn.Module, nn.Parameter, functional APIs (e.g., nn.functional.relu), and all layer types
# (e.g., nn.Linear, nn.Conv2d, nn.MaxPool2d) defined in the above code or in standard PyTorch.
# No previously instantiated layers (not even temporary ones inside your constructor) are allowed.
# The only allowed layer creation is the creation of one or more NEW layers for your own Net class
# (e.g., class Net < ...(nn.Module): def __init__(self)...(self, in_shape: tuple, out_shape: tuple, prm: dict) -> None:...).

# For example, the following is a valid output for this generator, which defines a 3-layer MLP with 1M parameters:
# class Net(nn.Module):
#     def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict) -> None:
#         super(Net, self).__init__()
#         self.flatten = nn.Flatten()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#         self.linear1 = nn.Linear(in_shape[1], 512)
#         self.linear2 = nn.Linear(512, 128)
#         self.linear3 = nn.Linear(128, out_shape[0])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.flatten(x)
#         x = self.relu(self.linear1(x))
#         x = self.dropout(x)
#         x = self.relu(self.linear2(x))
#         x = self.dropout(x)
#         x = self.linear3(x)
#         return x
# def supported_hyperparameters():
#     return ['lr', 'momentum']
# def train_setup(model, prm):
#     model.to(prm['device'])
#     model.train()
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=prm['lr'], momentum=prm['momentum'])
#     return model, loss_fn, optimizer
# def learn(model, data_loader, optimizer):
#     for inputs, labels in data_loader:
#         inputs, labels = inputs.to(prm['device']), labels.to(prm['device'])
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     model.to('cpu')

# If you feel the problem is too complex or does not contain enough information for a complete solution, please let me know and state your reasons.

# Note that the following code will be used to evaluate the network's performance on the training data:
# def test_net(net, prm, data_loader):
#     net.train_setup(prm)
#     learn(net, data_loader, net.optimizer)
#     return net.get_accuracy(data_loader)
# The network will be trained for at most one epoch. The final accuracy should be between 0.0 and 100.0.
# The training loop should include the following steps (and only these steps):
# 1. Initialize the network with the given hyperparameters (e.g., lr=0.01, momentum=0.9)
# 2. Load the data (inputs, labels) from the data_loader
# 3. Zero-out the gradients
# 4. Forward through the network to get the outputs
# 5. Compute the loss using a standard loss function (such as cross-entropy)
# 6. Backpropagate the gradients
# 7. Update the network weights
# 8. Move the network to the CPU and return the accuracy.
# The training loop should not include any other steps (such as validation or evaluation).
# The network should be moved to the CPU before returning the accuracy.
# The returned accuracy must be between 0.0 and 100.0. The returned accuracy is the best score achieved by the network on the training data after one epoch of training.
# The returned accuracy is the best score achieved by the network on the training data after one epoch of training.
# The returned accuracy is the best score achieved by the network on the training data after one epoch of training.

# The network must be defined in such a way that it can be trained with the given hyperparameters (e.g., lr=0.01, momentum=0.9).
# The network must be defined in such a way that it can be trained with the given hyperparameters (e.g., lr=0.01, momentum=0.9).
# The network must be defined in such a way that it can be trained with the given hyperparameters (e.g., lr=0.01, momentum=0.9).

# The network must be defined in such a way that it can be trained with the given hyperparameters (e.g.,