import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class ModuleWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl

class FlattenLayer(ModuleWrapper):
    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)

class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.conv2d(
            x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        act_var = 1e-16 + F.conv2d(
            x ** 2, self.W_sigma ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

supported_hyperparameters_dict = {'lr','momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.vocab_size = out_shape[0]
        self.hidden_dim = 640

        # Encoder: BBB Convolutional Network
        self.encoder = nn.Sequential(
            BBBConv2d(in_shape[1], 8, kernel_size=3, padding=1, bias=True, priors={
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-5, 0.1),
            }),
            nn.ReLU(),
            BBBConv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=True, priors={
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-5, 0.1),
            }),
            nn.ReLU(),
            BBBConv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=True, priors={
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-5, 0.1),
            }),
            nn.ReLU(),
            FlattenLayer(32),
            BBBLinear(32, self.hidden_dim, bias=True, priors={
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-5, 0.1),
            })
        )

        # Decoder: LSTM with embedding
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.fc_out = BBBLinear(self.hidden_dim, self.vocab_size, bias=True, priors={
            'prior_mu': 0,
            'prior_sigma': 0.1,
            'posterior_mu_initial': (0, 0.1),
            'posterior_rho_initial': (-5, 0.1),
        })

        # Initialize hyperparameters
        self.lr = float(prm.get('lr', 1e-3))
        self.momentum = float(prm.get('momentum', 0.9))

        # Store references for evaluation
        self.word2idx = prm.get('word2idx')
        self.idx2word = prm.get('idx2word')

    def init_zero_hidden(self, batch):
        return (
            torch.randn((2, batch, self.hidden_dim), device=self.device),
            torch.randn((2, batch, self.hidden_dim), device=self.device)
        )

    def train_setup(self, prm):
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    def learn(self, train_data):
        for images, captions in train_data:
            images = images.to(self.device)
            caps = captions.to(self.device) if captions.ndim == 3 else captions
            inputs = caps[:, :-1]
            targets = caps[:, 1:]
            memory = self.encoder(images)
            embedded = self.embedding(inputs)
            lstm_out, hidden = self.lstm(embedded)
            logits = self.fc_out(lstm_out)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.reshape(-1), label_smoothing=0.1)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

            if hasattr(self.scheduler, 'step') and callable(getattr(self.scheduler, 'step')):
                self.scheduler.step(loss)

    def forward(self, images, captions=None, hidden_state=None):
        images = images.to(self.device)
        memory = self.encoder(images)
        
        if captions is not None:
            captions = captions.to(self.device)
            if captions.ndim == 3:
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
            else:
                inputs = captions[:, :-1]
                targets = captions[:, 1:]
                targets = targets.contiguous()
                
            embedded = self.embedding(inputs)
            if hidden_state is None:
                lstm_out, hidden = self.lstm(embedded)
            else:
                lstm_out, hidden = self.lstm(embedded, hidden_state)
                
            logits = self.fc_out(lstm_out)
            
            return logits, hidden
    
    def kl_loss(self):
        kl_val = 0.0
        for name, module in self.named_modules():
            if isinstance(module, (BBBLinear, BBBConv2d)):
                kl_val += module.kl_loss()
        return kl_val

    def sample(self, images, max_length=20, temperature=1.0):
        pass

def supported_hyperparameters():
    return {'lr','momentum'}