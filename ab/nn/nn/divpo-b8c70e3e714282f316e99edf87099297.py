import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    def fuse(self):
        # Fused block for edge-friendly deployment
        fused = []
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                fused.append(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                fused.append(layer.weight)
        fused_conv = torch.nn.utils.fuse_conv_bn_eval(fused[0], fused[1])
        fused_block = Block(in_channels=fused_conv.in_channels,
                            out_channels=fused_conv.out_channels,
                            stride=self.conv[0].stride[0])
        fused_block.conv[0] = fused_conv
        fused_block.conv[1] = nn.Identity()
        fused_block.conv[2] = self.conv[2]
        return fused_block

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        in_channels = in_shape[1]  # derive from in_shape, not hardcode
        num_classes = int(out_shape[0])
        self.features = nn.Sequential(
            Block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(256, 512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Block(512, 512, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(512 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
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

    def get_fused_block(self):
        # Fused block for edge-friendly deployment
        fused = []
        for module in self.features.children():
            if isinstance(module, Block):
                fused.append(module.fuse())
            else:
                fused.append(module)
        return nn.Sequential(*fused)

    def deploy(self):
        # Generate edge-friendly version with fused blocks
        fused_features = self.get_fused_block()
        deploy_model = Net(
            in_shape=self.in_shape,
            out_shape=self.out_shape,
            prm=self.prm,
            device=self.device
        )
        deploy_model.features = fused_features
        deploy_model.classifier = self.classifier
        deploy_model.to(self.device)
        return deploy_model