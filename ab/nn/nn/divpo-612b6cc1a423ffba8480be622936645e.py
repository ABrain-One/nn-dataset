import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# #custom transformer layer
class TransLayer(nn.Module):
    def __init__(self, n_embed, n_head, attn_drop, proj_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_head, dropout=attn_drop)
        self.drop1 = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.GELU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x), return_dict=False)[0])
        x = x + self.mlp(self.norm2(x))
        return x


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, in_channels, n_embed, n_head, attn_drop, proj_drop, mlp_ratio=4.0):
        super().__init__()
        self.transformer = TransLayer(n_embed, n_head, attn_drop, proj_drop)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.LayerNorm(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.mlp_ratio = mlp_ratio
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x) + self.conv2(x)
        x = self.conv3(x)
        x = self.norm(x)
        return x


class Net(nn.Module):

    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        layers = []
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        layers += [
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        self.layer1 = Block(32, 32, n_head=2, attn_drop=0.1, proj_drop=0.1, mlp_ratio=4.0)
        layers += [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        self.layer2 = Block(64, 64, n_head=2, attn_drop=0.1, proj_drop=0.1, mlp_ratio=4.0)
        layers += [
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((6, 6))
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.features(self.stem(x))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=prm.get('lr', 0.01),
                                         momentum=prm.get('momentum', 0.9))

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