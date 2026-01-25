from torch.nn.functional import interpolate

from ab.nn.util.Const import *
from ab.nn.util.Train import *

def supported_hyperparameters():
    return {'lr'}  # batch, epoch, and transform are handled separately by the training framework


class RLFN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=46, mid_channels=48, upscale=4):
        super(RLFN, self).__init__()
        self.conv_input = torch.nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        
        self.block1 = RLFB(feature_channels, mid_channels)
        self.block2 = RLFB(feature_channels, mid_channels)
        self.block3 = RLFB(feature_channels, mid_channels)
        self.block4 = RLFB(feature_channels, mid_channels)
        
        self.conv_mid = torch.nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        
        self.upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(feature_channels, feature_channels * (upscale ** 2), kernel_size=3, padding=1),
            torch.nn.PixelShuffle(upscale),
            torch.nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out_fea = self.conv_input(x)
        out_1 = self.block1(out_fea)
        out_2 = self.block2(out_1)
        out_3 = self.block3(out_2)
        out_4 = self.block4(out_3)
        out_mid = self.conv_mid(out_4)
        out = self.upsampler(out_fea + out_mid)
        return out


class RLFB(torch.nn.Module):
    def __init__(self, in_channels, mid_channels=48):
        super(RLFB, self).__init__()
        self.c1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.c2 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.c3 = torch.nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.esa = ESA(in_channels, torch.nn.Conv2d)

    def forward(self, x):
        out = self.c1(x)
        out = torch.nn.functional.relu(out)
        out = self.c2(out)
        out = torch.nn.functional.relu(out)
        out = self.c3(out)
        out = self.esa(out)
        return out + x


class ESA(torch.nn.Module):
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
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = torch.nn.functional.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = torch.nn.functional.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class Net(torch.nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        
        # in_shape is (batch, channels, height, width) from get_in_shape
        # We need channels and spatial dimensions
        in_channels = in_shape[1] if len(in_shape) == 4 else in_shape[0]
        out_channels = out_shape[1] if len(out_shape) == 4 else out_shape[0]
        in_height = in_shape[2] if len(in_shape) == 4 else in_shape[1]
        out_height = out_shape[2] if len(out_shape) == 4 else out_shape[1]
        
        self.scale = out_height // in_height
        
        self.model = RLFN(in_channels=in_channels, out_channels=out_channels, upscale=self.scale)
        self.model.to(device)
        
        # Loss and Optimizer
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=prm.get('lr', 1e-3))
        
        self.best_score = 0.0

    def forward(self, x):
        return self.model(x)

    def train_setup(self, prm):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=prm.get('lr', 1e-3))

    def learn(self, data_roll: DataRoll):
        self.model.train()
        for inputs, labels in data_roll:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
