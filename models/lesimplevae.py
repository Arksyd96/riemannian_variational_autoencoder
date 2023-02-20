from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseVAE(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1, norm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.leaky_relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=0, output_padding=0, norm=True):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.leaky_relu(x)
        return x     

class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_channels, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        # layers
        self.in_conv = nn.Conv2d(input_shape[0], self.hidden_channels[0], kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList([
            Downsample(self.hidden_channels[i], self.hidden_channels[i+1], stride=2, padding=1)
            for i in range(len(self.hidden_channels) - 1)
        ])

        self.flatten = nn.Flatten()
        img_dim = (input_shape[1] // 2 ** (len(self.hidden_channels) - 1))
        self.latent_input_dim = (self.hidden_channels[-1], img_dim, img_dim)
        self.mu = nn.Linear(np.prod(self.latent_input_dim), self.latent_dim)
        self.logvar = nn.Linear(np.prod(self.latent_input_dim), self.latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.in_conv(x))
        for block in self.encoder:
            x = block(x)
        x = self.flatten(x)
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, latent_dim, latent_input_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels[::-1]
        self.latent_input_dim = latent_input_dim

        self.z_proj = nn.Linear(self.latent_dim, np.prod(latent_input_dim))

        self.decoder = nn.ModuleList([
            Upsample(self.hidden_channels[i], self.hidden_channels[i+1], stride=2)
            for i in range(len(self.hidden_channels) - 1)
        ])
        self.out = nn.Conv2d(self.hidden_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.z_proj(x))
        x = x.view(-1, *self.latent_input_dim)
        for block in self.decoder:
            x = block(x)
        x = self.out(x)
        return x

class LeSimpleVAE(nn.Module):
    def __init__(self, input_shape, out_channels, hidden_channels, latent_dim):
        super(LeSimpleVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_shape, hidden_channels, latent_dim).to(device)
        self.decoder = Decoder(out_channels, hidden_channels, latent_dim, self.encoder.latent_input_dim).to(device)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, eps, std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, eps, _ = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return dict(x=x, z=z, eps=eps, mu=mu, logvar=logvar)

    def loss_function(self, recons, input, mu, logvar, beta=1.):
        # recon_loss = F.binary_cross_entropy(recons, input, reduction='sum').div(input.size(0))
        # recon_loss = F.mse_loss(recons, input, reduction='mean')#.div(input.size(0))
        # kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # # kld_loss = kld_loss.div(input.size(0))
        recon_loss =  F.mse_loss(
            recons.reshape(input.shape[0], -1), input.reshape(input.shape[0], -1), reduction='none'
        ).sum()
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss * beta)

    def sample(self, n_samples):
        O = torch.normal(mean=0, std=1, size=(n_samples, self.latent_dim)).to(device)
        return self.decode(O)
