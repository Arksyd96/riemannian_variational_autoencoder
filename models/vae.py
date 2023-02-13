import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Decoder, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks, dec_blocks):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks).to(device)
        self.decoder = Decoder(out_channels, num_channels, latent_dim, bottleneck_ratio, dec_blocks).to(device)

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
        # reconstruction_loss = F.binary_cross_entropy(recons, input, reduction='sum')
        recon_loss = F.mse_loss(recons, input, reduction='sum').div(input.size(0))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss.div(input.size(0))
        return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss * beta)

    def sample(self, n_sample):
        z = torch.randn(n_sample, self.latent_dim).to(device)
        return self.decode(z)

    def _tempering(self, k, K):
        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    def log_p_x_given_z(self, recon_x, x, reduction='none'):
        return -F.binary_cross_entropy(
            recon_x, x, reduction=reduction
        ).reshape(x.shape[0], -1).sum(dim=1)

    def log_z(self, z):
        return self.normal.log_prob(z)

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling with q(z|x)
        """
        mu, log_var, features = self.encode(x)
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        recon_X = self.decode(Z, features if self.skip else None)

        recon = F.binary_cross_entropy(
            recon_X, x.repeat(sample_size, 1, 1, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -recon.reshape(sample_size, -1, self.input_shape).sum(dim=2)  # log(p(x|z))
        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))
        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)

        logpx = (logpxz + logpz - logqzx).logsumexp(dim=0).mean(dim=0) - torch.log(
            torch.Tensor([sample_size]).to(self.device)
        )
        return logpx

    def log_p_z_given_x(self, z, recon_x, x, sample_size=10):
        """
        Estimate log(p(z|x)) using Bayes rule and Importance Sampling for log(p(x))
        """
        logpx = self.log_p_x(x, sample_size)
        lopgxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return lopgxz + logpz - logpx

    def log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return logpxz + logpz

    ########## Kullback-Leiber divergences estimates ##########

    def kl_prior(self, mu, log_var):
        """KL[q(z|y) || p(z)] : exact formula"""
        # return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_cond(self, recon_x, x, z, mu, log_var, sample_size=10):
        """
        KL[p(z|x) || q(z|x)]
        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, recon_x, x, sample_size=sample_size)
        logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(z)

        return (logqzx - logpzx).sum()

    def sample_img(
        self,
        z=None,
        x=None,
        step_nbr=1,
        record_path=False,
        n_samples=1,
        verbose=False,
    ):
        """
        Simulate p(x|z) to generate an image
        """
        assert self.skip == False or x is not None, "x must be provided if skip connections are used"
        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        if x is not None:
            recon_x, z, _, _, _, features = self.forward(x)

        z.requires_grad_(True)
        recon_x = self.decode(z, features if self.skip else None)
        return recon_x