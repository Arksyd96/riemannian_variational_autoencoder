from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class EncodingBlock(nn.Module):
    def __init__(self, num_channels, bottleneck_ratio, residual=True, norm=True, downrate=None, res=None):
        super(EncodingBlock, self).__init__()
        self.downrate = downrate
        self.residual = residual
        self.res = res

        # layers
        mid_channels = int(num_channels * bottleneck_ratio)
        self.conv_a = nn.Conv2d(num_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_d = nn.Conv2d(mid_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.norm = nn.BatchNorm2d(num_channels) if norm else None

    def forward(self, x):
        if self.downrate is not None:
            x = F.max_pool2d(x, kernel_size=self.downrate, stride=self.downrate, padding=0)
        if self.residual:
            residual = x
        x = F.gelu(self.conv_a(x))
        x = F.gelu(self.conv_b(x))
        x = F.gelu(self.conv_c(x))
        x = self.conv_d(x)
        if self.norm:
            x = self.norm(x)
        if self.residual:
            x += residual
        x = F.gelu(x)
        return x

class DecodingBlock(nn.Module):
    def __init__(self, num_channels, bottleneck_ratio, residual=True, norm=True, uprate=None, res=None):
        super(DecodingBlock, self).__init__()
        self.uprate = uprate
        self.residual = residual
        self.res = res

        # layers
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_u = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        mid_channels = int(num_channels * bottleneck_ratio)
        self.conv_a = nn.Conv2d(num_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_d = nn.Conv2d(mid_channels, num_channels, kernel_size=1, stride=1, padding=0)   
        self.norm = nn.BatchNorm2d(num_channels) if norm else None    

    def forward(self, x):
        if self.uprate is not None:
            x = self.up(x)
            x = F.gelu(self.conv_u(x))

        # input is CHW
        # diffY = self.res - x.size(2)
        # diffX = self.res - x.size(3) 

        # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])

        if self.residual:
            residual = x
        x = F.gelu(self.conv_a(x))
        x = F.gelu(self.conv_b(x))
        x = F.gelu(self.conv_c(x))
        x = self.conv_d(x)
        if self.norm:
            x = self.norm(x)
        if self.residual:
            x += residual
        x = F.gelu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels, latent_dim, bottleneck_ratio, enc_blocks):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.enc_blocks = enc_blocks
        self.bottleneck_ratio = bottleneck_ratio
        
        # layers
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList([
            EncodingBlock(num_channels, bottleneck_ratio, residual=True, norm=True, downrate=downrate, res=res)
            for res, downrate in self.enc_blocks
        ])

        self.flatten = nn.Flatten()
        latent_input_dim = num_channels * self.enc_blocks[-1][0] ** 2 
        self.mu = nn.Linear(latent_input_dim, self.latent_dim)
        self.logvar = nn.Linear(latent_input_dim, self.latent_dim)

    def forward(self, x):
        x = F.gelu(self.in_conv(x))
        for block in self.encoder:
            x = block(x)
        x = self.flatten(x)
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, num_channels, latent_dim, bottleneck_ratio, dec_blocks):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.bottleneck_ratio = bottleneck_ratio
        self.dec_blocks = dec_blocks
        self.num_channels = num_channels

        latent_output_dim = num_channels * dec_blocks[0][0] ** 2
        self.z_proj = nn.Linear(self.latent_dim, latent_output_dim)

        self.decoder = nn.ModuleList([
            DecodingBlock(num_channels, bottleneck_ratio, residual=True, norm=True, uprate=uprate, res=res)
            for res, uprate in self.dec_blocks
        ])
        self.out = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.gelu(self.z_proj(x))
        x = x.view(-1, self.num_channels, self.dec_blocks[0][0], self.dec_blocks[0][0])
        for block in self.decoder:
            x = block(x)
        x = torch.sigmoid(self.out(x))
        return x