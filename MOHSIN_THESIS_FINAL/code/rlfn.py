import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- 1. The ESA Block (Enhanced Spatial Attention) ---
# This is the "Secret Sauce" of RLFN. It makes the network focus on 
# edges and textures rather than flat backgrounds.
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

# --- 2. The RLFB Block (Residual Local Feature Block) ---
# This combines 3 convolutions with the ESA attention mechanism.
class RLFB(nn.Module):
    def __init__(self, n_feats, mid_feats=None):
        super(RLFB, self).__init__()
        mid_feats = mid_feats or n_feats
        self.c1 = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        self.c2 = nn.Conv2d(mid_feats, mid_feats, 3, 1, 1)
        self.c3 = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)
        self.esa = ESA(n_feats, nn.Conv2d)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        y = self.c1(x)
        y = self.act(y)
        y = self.c2(y)
        y = self.act(y)
        y = self.c3(y)
        y = self.esa(y)
        return y + x

# --- 3. The Main Network (RLFN) ---
class Net(nn.Module):
    def train_setup(self, prm):
        # We tune the Learning Rate based on Optuna's suggestion
        self.optimizer = optim.Adam(self.parameters(), lr=prm['lr'])

    def learn(self, train_data):
        # The custom training loop required by the framework
        criterion = nn.L1Loss()
        device = next(self.parameters()).device
        
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            self.optimizer.zero_grad()
            output = self(inputs)
            loss = criterion(output, labels)
            loss.backward()
            self.optimizer.step()

    def __init__(self, in_shape=(3, 64, 64), out_shape=(3, 256, 256), prm=None, *args, **kwargs):
        super(Net, self).__init__()
        
        # Configuration
        n_feats = 52       # Number of channels (standard for RLFN)
        n_blocks = 4       # Number of blocks (standard is 4 or 6)
        scale = 4          # Upscaling factor (64 -> 256 is 4x)
        
        # 1. Shallow Feature Extraction
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        
        # 2. Deep Feature Extraction (The RLFBs)
        self.body = nn.Sequential(*[RLFB(n_feats) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        
        # 3. Reconstruction (Upsampling)
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, 3 * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.body_tail(res)
        res += x
        x = self.tail(res)
        return x

def supported_hyperparameters():
    return {'lr'}
