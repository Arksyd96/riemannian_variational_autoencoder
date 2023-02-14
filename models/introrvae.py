import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rvae import RVAE
from .new_base import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IntroRVAE(RVAE):
    def __init__(self, in_channels, out_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks, dec_blocks, dt):
        RVAE.__init__(self, in_channels, out_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks, dec_blocks, dt)
        
        # transform the encoder into a discriminator
        self.discriminator = Encoder(in_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks)
        self.discriminator.logvar = nn.Identity()
        input_dim = self.discriminator.mu.in_features
        self.discriminator.mu = nn.Linear(input_dim, 1)

    def forward_discriminator(self, x):
        return torch.sigmoid(self.discriminator(x))

    def loss_function(self, recons, input, fake, mu, logvar, beta=1.):
        recon_loss = F.mse_loss(recons, input, reduction='sum').div(input.size(0))
        kld_loss = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1)
        kld_loss = kld_loss.div(input.size(0))
        d_loss_real = F.binary_cross_entropy(input, torch.ones_like(fake))
        d_loss_fake = F.binary_cross_entropy(fake, torch.zeros_like(fake))
        d_loss = d_loss_real + d_loss_fake
        loss = recon_loss + beta * kld_loss + d_loss
        return dict(loss=loss, recon_loss=recon_loss, kld_loss=kld_loss * beta, d_loss=d_loss)
