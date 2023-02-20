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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        # layers
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.double_conv = DoubleConv(out_channels, out_channels)


    def forward(self, x):
        x = F.relu(self.down_conv(x))
        x = self.bn(x)
        x = self.double_conv(x)
        return x

class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodingBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.double_conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.up_conv(x))
        x = self.bn(x)
        x = self.double_conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, image_shape, latent_dim, hidden_channels):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        # layers
        self.in_conv = nn.Conv2d(image_shape[0], hidden_channels[0], kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList([
            EncodingBlock(hidden_channels[i], hidden_channels[i + 1])
            for i in range(len(hidden_channels) - 1)
        ])

        self.flatten = nn.Flatten()
        self.image_shape = image_shape[-1] // (2 ** (len(hidden_channels) - 1))
        latent_input_dim = self.hidden_channels[-1] * self.image_shape ** 2 
        self.mu = nn.Linear(latent_input_dim, self.latent_dim)
        self.logvar = nn.Linear(latent_input_dim, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.in_conv(x))
        for block in self.encoder:
            x = block(x)
        x = self.flatten(x)
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, image_shape, out_channels, latent_dim, hidden_channels):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        self.image_shape = image_shape[-1] // (2 ** (len(hidden_channels) - 1))
        latent_output_dim = self.hidden_channels[0] * self.image_shape ** 2
        self.z_proj = nn.Linear(self.latent_dim, latent_output_dim)

        self.decoder = nn.ModuleList([
            DecodingBlock(self.hidden_channels[i], self.hidden_channels[i + 1])
            for i in range(len(self.hidden_channels) - 1)
        ])
        self.out = nn.Conv2d(self.hidden_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.z_proj(x))
        x = x.view(-1, self.hidden_channels[0], self.image_shape, self.image_shape)
        for block in self.decoder:
            x = block(x)
        x = self.out(x)
        return x