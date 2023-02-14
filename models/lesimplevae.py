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
        x = torch.sigmoid(self.out(x))
        return x

class LeSimpleVAE(nn.Module):
    def __init__(self, input_shape, out_channels, hidden_channels, latent_dim, dt):
        super(LeSimpleVAE, self).__init__()
        self.latent_dim = latent_dim
        self.dt = dt

        self.encoder = Encoder(input_shape, hidden_channels, latent_dim).to(device)
        self.decoder = Decoder(out_channels, hidden_channels, latent_dim, self.encoder.latent_input_dim).to(device)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = self.hyperboloide_normal_sample(mu, self.dt)
        return mu + eps * std, eps, std

    def hyperboloide_normal_sample(self, mu, dt):
        dM = mu  * torch.normal(mean=0, std=np.sqrt(dt), size=mu.shape).to(device)
        dSigma = (mu / np.sqrt(2)) * torch.normal(mean=0, std=np.sqrt(dt), size=mu.shape).to(device)
        return dM + torch.randn(mu.shape).to(device) * dSigma

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, eps, _ = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return dict(x=x, z=z, eps=eps, mu=mu, logvar=logvar)

    def loss_function(self, recons, input, mu, logvar, beta=1.):
        # recon_loss = F.binary_cross_entropy(recons, input, reduction='sum').div(input.size(0))
        recon_loss = F.mse_loss(recons, input, reduction='sum').div(input.size(0))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss.div(input.size(0))
        return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss * beta)

    def sample(self, n_samples):
        O = torch.normal(mean=0, std=1, size=(n_samples, self.latent_dim)).to(device)
        O = self.hyperboloide_normal_sample(O, self.dt)
        return self.decode(O)

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
        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        if x is not None:
            recon_x, z, _, _, _, features = self.forward(x)

        z.requires_grad_(True)
        recon_x = self.decode(z)
        return recon_x