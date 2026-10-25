import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}


class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]  # This should be 3 for RGB images
        self.num_classes = out_shape[0]
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        self.dropout_rate = prm.get('dropout', 0.5)

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
            self.optimizer.step()

    def allocate_and_copy(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Allocating memory for output tensor
        output_tensor = torch.empty_like(input_tensor)

        # Copying input tensor to output tensor
        output_tensor.copy_(input_tensor)

        return output_tensor

    def forward_with_memory_optimization(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Preprocessing input tensor
        preprocessed_input = self.preprocess(input_tensor)

        # Forward pass through the network with memory optimization
        memory_optimized_output = self.forward(preprocessed_input)

        # Postprocessing output tensor
        postprocessed_output = self.postprocess(memory_optimized_output)

        return postprocessed_output

    def preprocess(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Resize input tensor to network input size
        preprocessed_input = F.interpolate(input_tensor, size=(32, 32))

        # Normalize input tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocessed_input = F.normalize(preprocessed_input, mean=mean, std=std)

        return preprocessed_input

    def postprocess(self, output_tensor: torch.Tensor) -> torch.Tensor:
        # Apply softmax to output tensor
        postprocessed_output = F.softmax(output_tensor, dim=1)

        return postprocessed_output