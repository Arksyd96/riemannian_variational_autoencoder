import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RVAE(VAE):
    def __init__(self, in_channels, out_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks, dec_blocks, dt):
        VAE.__init__(self, in_channels, out_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks, dec_blocks)
        assert latent_dim % 2 == 0, "Latent dimension must be even"
        self.dt = dt

        in_features = self.encoder.logvar.in_features
        self.encoder.logvar = nn.Linear(in_features, 1)
        self.encoder.mu = nn.Identity()

        self.m = nn.Linear(in_features, latent_dim // 2, bias=True)
        self.sigma = nn.Linear(in_features, latent_dim // 2, bias=True)

        self.mean_h, self.std_h = [], []
        self.mean, self.std = 0, 1

    def forward(self, x):
        x, S = self.encode(x)
        m, sigma = self.m(x), self.sigma(x)
        S = torch.tanh(S) + 1 # between 0 and 2
        z = self.manifold_projection(m, sigma, S)
        if self.training:
            self.mean_h.append(z.mean())
            self.std_h.append(z.std())
        x = self.decode(z)
        return dict(x=x, z=z, m=m, sigma=sigma, S=S)

    def manifold_projection(self, m, sigma, S):
        batch = m.size(0)
        n_steps = torch.floor(S / self.dt).to(torch.int8)
        for b in range(batch):
            m[b, :], sigma[b, :] = self.brownian_step(m[b, :], sigma[b, :], n_steps[b])
        return torch.cat([m, sigma], dim=1)
        
    def brownian_step(self, m, sigma, n_steps):
        std = np.sqrt(self.dt)
        for _ in range(n_steps):
            m = m + sigma * torch.normal(mean=0, std=std, size=sigma.size()).to(device)
            sigma = sigma + (sigma / np.sqrt(2)) * torch.normal(mean=0, std=std, size=sigma.size()).to(device)
        return m, sigma

    def update(self):
        self.mean = torch.tensor(self.mean_h).mean()
        self.std = torch.tensor(self.std_h).mean()
        self.mean_h, self.std_h = [], []

    def loss_function(self, recons, input, mu, logvar, beta=1.):
        # recon_loss = F.binary_cross_entropy(recons, input, reduction='sum').div(input.size(0))
        recon_loss = F.mse_loss(recons, input, reduction='sum').div(input.size(0))
        # kld_loss = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1)# self.dt + np.log(self.dt))
        # kld_loss = kld_loss.div(input.size(0))
        # return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss * beta)
        return dict(loss=recon_loss, recon_loss=recon_loss, kld_loss=0)

    def sample(self, n_samples, n_steps=10):
        O = torch.normal(mean=self.mean, std=self.std, size=(n_samples, self.latent_dim)).to(device)
        m, sigma = O[:, :self.latent_dim // 2], O[:, self.latent_dim // 2:]
        m, sigma = self.brownian_step(m, sigma, n_steps)
        O = torch.cat([m, sigma], dim=1)
        return self.decode(O)