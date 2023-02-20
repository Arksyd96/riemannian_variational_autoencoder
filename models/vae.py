import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Decoder, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_shape, out_channels, latent_dim, hidden_channels):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_shape, latent_dim, hidden_channels).to(device)
        self.decoder = Decoder(input_shape, out_channels, latent_dim, hidden_channels[::-1]).to(device)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, eps, std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, eps, _ = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return dict(x=x, z=z, eps=eps, mu=mu, logvar=logvar)

    def loss_function(self, recons, input, mu, logvar, beta=1.):
        # # recon_loss = F.binary_cross_entropy(recons, input, reduction='sum').div(input.size(0))
        # recon_loss = F.mse_loss(recons, input, reduction='mean')
        # kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss =  F.mse_loss(recons, input, reduction='none').sum()
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss * beta)

    def sample(self, n_sample):
        z = torch.randn(n_sample, self.latent_dim).to(device)
        return self.decode(z)