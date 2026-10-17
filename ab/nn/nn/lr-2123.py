import torch
import torch.nn as nn
import torch.nn.functional as F

def supported_hyperparameters():
    return {'epoch_max', 'gamma', 'lr', 'momentum', 'step_size', 'weight_decay'}
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, t):
        x = x.real if torch.is_complex(x) else x
        
        # Time embedding
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        t_emb = t_emb.view(t_emb.shape[0], -1, 1, 1)
        
        # Feature extraction
        x = self.encoder(x)
        x = x + t_emb
        
        # Classification
        return self.classifier(x)
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.timesteps = 100
        self.channels = in_shape[1]
        self.num_classes = out_shape[0]
        # UNet for diffusion and classification
        self.unet = UNet(self.channels, self.num_classes)
        # Initialize diffusion parameters
        self.beta = torch.linspace(0.00001, 0.01, self.timesteps).to(device) 
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def forward(self, x):
        if self.training:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device)
            x = self.add_noise(x, t)
        else:
            t = torch.zeros(x.shape[0], device=self.device)
        return self.unet(x, t)
        
    def add_noise(self, x, t):
        noise = torch.randn_like(x).to(self.device)
        alpha_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'], weight_decay=prm.get('weight_decay', 1e-05))
        step_size = int(prm.get('step_size', 0.1) * prm.get('epoch_max', 50))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=prm.get('gamma', 0.1))
    
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
        self.scheduler.step()