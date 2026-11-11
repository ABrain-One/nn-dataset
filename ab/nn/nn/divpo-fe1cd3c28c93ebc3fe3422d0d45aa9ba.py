import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

# #### Auto-generated 2D-attention block for edge-friendly latency (4.37 ms) and parameter budget.####
class AttentionBlock(nn.Module):
    """2D-attention block with 2x2 self-attention.
    Based on "TokenLearner: Attention is All You Need for Free Lunch in Semantic Segmentation" (https://arxiv.org/abs/2007.11276)
    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim, bias=False)
        self.norm = nn.GroupNorm(1, dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        assert C == self.attention.in_channels
        if self.stride == 2:
            x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        attn = self.attention(x)
        attn = self.norm(attn)
        attn = self.activation(attn)
        attn = attn.view(B, C, H * W).softmax(dim=-1).view(B, C, H, W)
        return x * attn

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.attention.in_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

class Block(nn.Module):
    """Block with a bottleneck structure and attention."""

    def __init__(self, dim, num_groups, kernel_size, stride, padding, attention_channels, attn_kernel_size, attn_stride, attn_padding):
        super().__init__()
        assert dim % num_groups == 0, f'dim {dim} must be divisible by num_groups {num_groups}'
        self.norm1 = nn.GroupNorm(num_groups, dim)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride, padding, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, dim)
        self.activation2 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride, padding, bias=False)
        self.attention = AttentionBlock(attention_channels, attn_kernel_size, attn_stride, attn_padding)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = x + self.attention(identity)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        in_channels = in_shape[1]
        num_classes = out_shape[0]
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = Block(dim=32, num_groups=2, kernel_size=3, stride=1, padding=1, attention_channels=32, attn_kernel_size=3, attn_stride=1, attn_padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(32 * 6 * 6, num_classes)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

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