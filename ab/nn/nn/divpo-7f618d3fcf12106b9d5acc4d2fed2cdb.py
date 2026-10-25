import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

# Auto-generated single-layer block for reference
class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Target function for the new model
def generate_model(in_shape, out_shape, prm):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Derive the first conv's input channels based on the input shape
    in_channels = in_shape[1]
    
    # Define the class with the new block and prm for IDE compatibility
    class NetWithBlock(Net):
        def __init__(self, in_shape, out_shape, prm, device):
            super().__init__(in_shape, out_shape, prm, device)
            self.block = Block(self.in_channels)  # Use the provided Block
            self.block2 = Block(64)  # Example block with different in_channels
        
        def forward(self, x):
            x = self.block(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.block2(x)  # Apply the new block
            
            x = self.layer2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Create the model instance for shape and parameter inference
    model = NetWithBlock(in_shape, out_shape, prm, device)
    prm['lr'] = 0.01  # Ensure the learning rate is defined
    prm['momentum'] = 0.9  # Ensure momentum is defined
    in_channels = in_shape[1]
    channels = [32, 64, 128]
    num_classes = out_shape[0]
    model_size = 0
    
    # Target size bound: 500000
    # Design the model to meet or slightly exceed the target size limit.
    # The block should contribute to the size estimation.
    
    # Example of a valid layer (you must design a new one based on the block):
    # block = Block(in_channels, channels[0], num_classes)
    # model_size += sum(p.numel() for p in block.parameters())
    
    # This is a template for the forward method of the designed block:
    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     return x
    
    # Design the NetWithBlock class to use the new block, keep the total
    # number of parameters below 500000, and ensure it conforms to the
    # provided in_shape and adheres to the resource constraints.

    # The target for the code execution is to return a valid Python class
    # (not an instance of the class) that defines the NetWithBlock class
    # with the new block. The class must be defined inside the function
    # with the same name "NetWithBlock" and prefixed with "NetWithBlock.".
    # E.g., def NetWithBlock(in_shape, out_shape, prm):
    #                                               ^--- This is the target name.
    # Inside the function, you can define the helper block class and then
    # return the new class definition at the end (do not return the class
    # instance or any other non-class object).
    # The class must be complete and correctly indented within the function.
    # The class must be the ONLY object you define and return.
    return NetWithBlock