import numpy as np
import torch
import torch.nn.functional as F
from .vae import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RVAE(VAE):
    def __init__(self, input_shape, out_channels, latent_dim, hidden_channels, dt, skip, device='cuda'):
        VAE.__init__(self, input_shape, out_channels, latent_dim, hidden_channels, skip, device)
        # step size
        self.dt = dt

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = self.hyperboloide_normal_sample(mu, self.dt)
        return mu + eps * std, eps, std

    def hyperboloide_normal_sample(self, mu, dt):
        noise = torch.normal(mean=0, std=np.sqrt(dt), size=mu.shape).to(device)
        Y = torch.zeros_like(mu).to(device)
        Y = (mu / 4) * dt + (mu / np.sqrt(2)) * noise
        return Y

    def loss_function(self, recons, input, mu, logvar, beta=1.):
        recon_loss = F.mse_loss(recons, input, reduction='sum').div(input.size(0))
        kld_loss = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + self.dt + np.log(self.dt))
        kld_loss = kld_loss.div(input.size(0))
        return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss * beta)

    def sample(self, n_samples, dt):
        O = torch.normal(mean=0, std=np.sqrt(dt), size=(n_samples, self.latent_dim)).to(device)
        O = self.hyperboloide_normal_sample(O, self.dt)
        return self.decode(O, None)