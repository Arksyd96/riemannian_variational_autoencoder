import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_enc_string(s):
    blocks = []
    for block in s.split(','):
        if 'x' in block:
            res, num = block.split('x')
            blocks.extend([(int(res), None) for _ in range(int(num))])
        elif 'd' in block:
            res, down_rate = block.split('d')
            blocks.append((int(res), int(down_rate)))
        elif 'u' in block:
            res, up_rate = block.split('u')
            blocks.append((int(res), int(up_rate)))
    return blocks

def weight_norm(module):
    module = nn.utils.weight_norm(module, dim=None)
    return module

class EncodingBlock(nn.Module):
    def __init__(self, res, num_channels, bottleneck_ratio, kernel_size, residual=True, down_rate=None):
        super(EncodingBlock, self).__init__()
        self.residual = residual
        self.down_rate = down_rate
        self.res = res

        # convs
        mid_channels = int(num_channels * bottleneck_ratio)
        self.conv_a = nn.Conv2d(num_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_d = nn.Conv2d(mid_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        if self.down_rate is not None:
            x = F.avg_pool2d(x, kernel_size=self.down_rate, stride=self.down_rate)
        if self.residual:
            residual = x
        x = F.gelu(self.conv_a(x))
        x = F.gelu(self.conv_b(x))
        x = F.gelu(self.conv_c(x))
        x = self.conv_d(x)
        if self.residual:
            x = x + residual
        x = self.bn(x)
        x = F.gelu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels, enc_blocks, bottleneck_ratio) -> None:
        super().__init__()
        self.bottleneck_ratio = bottleneck_ratio
        self.enc_blocks = parse_enc_string(enc_blocks)
        self.n_blocks = self.enc_blocks.__len__()

        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder = nn.ModuleList([
            EncodingBlock(res, num_channels, bottleneck_ratio, 3, residual=True, down_rate=down_rate) 
            for res, down_rate in self.enc_blocks
        ])

    def forward(self, x):
        x = F.gelu(self.in_conv(x))
        for block in self.encoder:
            x = block(x)
        return x

class DecodingBlock(nn.Module):
    def __init__(self, res, num_channels, bottleneck_ratio, kernel_size, residual=True, up_rate=None):
        super(DecodingBlock, self).__init__()
        self.residual = residual
        self.up_rate = up_rate
        self.res = res

        # learnable parameters
        mid_channels = int(num_channels * bottleneck_ratio)
        self.conv_a = nn.Conv2d(num_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_d = nn.Conv2d(mid_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(num_channels)


    def forward(self, x):
        if self.up_rate is not None:
            x = F.interpolate(x, scale_factor=self.up_rate, mode='nearest')
        if self.residual:
            residual = x
        x = F.gelu(self.conv_a(x))
        x = F.gelu(self.conv_b(x))
        x = F.gelu(self.conv_c(x))
        x = self.conv_d(x)
        if self.residual:
            x = x + residual
        x = self.bn(x)
        x = F.gelu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, num_channels, dec_blocks, bottleneck_ratio) -> None:
        super().__init__()
        self.bottleneck_ratio = bottleneck_ratio
        self.dec_blocks = parse_enc_string(dec_blocks)
        self.n_blocks = self.dec_blocks.__len__()

        # blocks
        self.decoder = nn.ModuleList([
            DecodingBlock(res, num_channels, bottleneck_ratio, 3, residual=True, up_rate=up_rate) 
            for res, up_rate in self.dec_blocks
        ])
        self.out_conv = nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        for block in self.decoder:
            x = block(x)
        x = torch.sigmoid(self.out_conv(x))
        return x

class DVAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, num_channels, enc_blocks, dec_blocks, bottleneck_ratio, riemannian=False, dt=0.1) -> None:
        super().__init__()
        self.dt = dt
        self.latent_dim = latent_dim
        self.riemannian = riemannian

        self.encoder = Encoder(in_channels, num_channels, enc_blocks, bottleneck_ratio)
        self.decoder = Decoder(out_channels, num_channels, dec_blocks, bottleneck_ratio)

        assert self.encoder.n_blocks == self.decoder.n_blocks, "Encoder and Decoder must have the same number of blocks"

        self.mu_projection = nn.Conv2d(num_channels, self.latent_dim, kernel_size=3, stride=1, padding=1)
        self.logvar_projection = nn.Conv2d(num_channels, self.latent_dim, kernel_size=3, stride=1, padding=1)
        self.latent_projection = weight_norm(nn.Conv2d(latent_dim, num_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if not self.riemannian:
            eps = torch.randn_like(std)
        else:
            eps = self.hyperboloide_normal_sample(mu, dt=self.dt)
        return mu + eps * std

    def hyperboloide_normal_sample(self, mu, dt):
        noise = torch.normal(mean=0, std=np.sqrt(dt), size=mu.shape).to(device)
        Y = torch.zeros_like(mu).to(device)
        Y = (mu / 4) * dt + (mu / np.sqrt(2)) * noise
        return Y

    def forward(self, x):
        x = self.encoder(x)
        # mu, logvar = self.latent_projection(x).chunk(2, dim=1)
        mu, logvar = self.mu_projection(x), self.logvar_projection(x)
        z = self.reparametrize(mu, logvar)
        x = self.latent_projection(z)
        x = self.decoder(x)
        return dict(x=x, z=z, mu=mu, logvar=logvar)

    def loss_function(self, x_hat, x, mu, logvar, beta=1.):
        recon_loss = F.mse_loss(x_hat, x, reduction='none').sum(dim=(1, 2, 3)).mean()
        kld_loss = self.kl_divergence(mu, logvar).div(x.shape[0])
        return dict(loss=recon_loss + beta * kld_loss, recon_loss=recon_loss, kld_loss=kld_loss) 

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1)

    def sample(self, n_samples, dt):
        B = n_samples
        features_dim = self.decoder.dec_blocks[0][0]
        z = torch.normal(mean=0, std=1, size=(B, self.latent_dim, features_dim, features_dim)).to(device)
        if self.riemannian:
            z = self.hyperboloide_normal_sample(z, dt=dt)
        x = self.latent_projection(z)
        x = self.decoder(x)
        return x