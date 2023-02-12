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
        # layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.stride = stride
        
        mid_channels = int(out_channels * 0.5)
        self.conv_a = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x = self.conv(x)
        # if self.norm:
        #     x = self.norm(x)
        # x = self.leaky_relu(x)
        
        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        
        x = self.leaky_relu(self.conv_a(x))
        x = self.leaky_relu(self.conv_b(x))
        x = self.leaky_relu(self.conv_c(x))
        x = self.leaky_relu(self.conv_d(x))
        x = self.norm(x) if self.norm else x
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, skip=False, norm=True):
        super(Upsample, self).__init__()
        self.output_shape = output_shape
        self.skip = skip

        if self.skip:
            self.mid_channels = in_channels * 2 if in_channels == out_channels else in_channels
        else:
            self.mid_channels = out_channels

        # layers
        # self.up = nn.ConvTranspose2d(
        #     in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0
        # )
        self.conv = nn.Conv2d(self.mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2, padding=0, output_padding=0
        )
        mid_channels = int(out_channels * 0.5)
        self.conv_a = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x, features=None):
        x = self.up(x)

        # input is CHW
        diffY = self.output_shape[0] - x.size(2)
        diffX = self.output_shape[1] - x.size(3)

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if self.skip:
            x = torch.cat([x, features], dim=1)

        # x = self.conv(x)
        # if self.norm:
        #     x = self.norm(x)

        x = self.leaky_relu(self.conv_a(x))
        x = self.leaky_relu(self.conv_b(x))
        x = self.leaky_relu(self.conv_c(x))
        x = self.leaky_relu(self.conv_d(x))
        x = self.norm(x) if self.norm else x
        return x

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, hidden_channels):
        super(Encoder, self).__init__()
        assert input_shape.__len__() >= 3, 'input_shape must be (C, H, W, D?)'
        self.input_shape = np.array(input_shape)
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        self.downsamples = nn.ModuleList(
            [Downsample(self.input_shape[0], self.hidden_channels[0], stride=1, norm=False)] +
            [Downsample(hidden_channels[i], hidden_channels[i + 1]) for i in range(hidden_channels.__len__() - 1)]
        )

        self.encoding_shapes, shape = list(), self.input_shape[1:]
        self.encoding_shapes.append(shape)
        for i in range(hidden_channels.__len__() - 1):
            shape = (shape + 1) // 2
            self.encoding_shapes.append(shape)

        self.flatten = nn.Flatten()
        self.mu = nn.Linear(self.hidden_channels[-1] * np.prod(self.encoding_shapes[-1]), self.latent_dim)
        self.log_var = nn.Linear(self.hidden_channels[-1] * np.prod(self.encoding_shapes[-1]), self.latent_dim)

    def forward(self, x):
        features = list()
        for downsampler in self.downsamples:
            features.append(x)
            x = downsampler(x)
        x = self.flatten(x)
        return self.mu(x), self.log_var(x), features

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, hidden_channels, skip, encoding_shapes):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.encoding_shapes = encoding_shapes

        self.fully_connected = nn.Linear(
            self.latent_dim, self.hidden_channels[0] * np.prod(self.encoding_shapes[0])
        )

        # bad idea to normalize at input/output layers
        self.upsamples = nn.ModuleList(
            [
                Upsample(self.hidden_channels[i], self.hidden_channels[i + 1], self.encoding_shapes[i + 1], skip=skip)
                for i in range(self.hidden_channels.__len__() - 1)
            ]
        )
        self.out = nn.Conv2d(self.hidden_channels[-1], self.out_channels, kernel_size=1)

    def forward(self, x, features):
        x = self.fully_connected(x)
        x = x.view(-1, self.hidden_channels[0], *self.encoding_shapes[0])
        for idx, upsampler in enumerate(self.upsamples):
            x = upsampler(x, features[features.__len__() - 1 - idx] if features != None else None)
        return torch.sigmoid(self.out(x))