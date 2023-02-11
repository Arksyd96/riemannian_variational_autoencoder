import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import itertools
from torchsummary import summary

@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))

def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)

class DmolNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.H.dataset in ['ffhq_256'])

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers

class Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, down_rate=None, residual=False, use_3x3=True) -> None:
        super().__init__()
        self.residual = residual
        self.down_rate = down_rate
        
        # conv
        self.conv_a = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3 if use_3x3 else 1, stride=1, padding=(1 if use_3x3 else 0))
        self.conv_c = nn.Conv2d(mid_channels, mid_channels, kernel_size=3 if use_3x3 else 1, stride=1, padding=(1 if use_3x3 else 0))
        self.conv_d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        Fx = self.conv_a(F.gelu(x))
        Fx = self.conv_b(F.gelu(Fx))
        Fx = self.conv_c(F.gelu(Fx))
        Fx = self.conv_d(F.gelu(Fx))
        if self.residual:
            Fx = Fx + x
        if self.down_rate is not None:
            Fx = F.avg_pool2d(Fx, kernel_size=self.down_rate, stride=self.down_rate)
        return Fx
        
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, enc_blocks, bottleneck_ratio) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
        blockstr = parse_layer_string(enc_blocks)
        self.encoder = nn.ModuleList(
            [
                Block(
                    out_channels, 
                    int(out_channels * bottleneck_ratio), 
                    out_channels, 
                    down_rate=down, 
                    residual=True, 
                    use_3x3=(True if res > 2 else False)
                ) for res, down in blockstr
            ]
        )
        
        self.n_blocks = len(blockstr)
        for block in self.encoder:
            block.conv_d.weight.data *= np.sqrt(1 / self.n_blocks)
            
    def forward(self, x):
        activations = {}
        x = self.in_conv(x)
        activations[x.shape[2]] = x
        for block in self.encoder:
            x = block(x)
            activations[x.shape[2]] = x
        return activations
    
class DecoderBlock(nn.Module):
    def __init__(self, latent_dim, res, num_channels, bottleneck_ratio, n_blocks, mixin=None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.res = res
        
        self.encoding_block = Block(
            in_channels=num_channels * 2,
            mid_channels=int(num_channels * bottleneck_ratio),
            out_channels=self.latent_dim * 2,
            residual=False,
            use_3x3=res > 2
        )
        self.prior = Block(
            in_channels=num_channels,
            mid_channels=int(num_channels * bottleneck_ratio),
            out_channels=self.latent_dim * 2 + num_channels,
            residual=False,
            use_3x3=res > 2
        )
        self.resnet = Block(
            in_channels=num_channels,
            mid_channels=int(num_channels * bottleneck_ratio),
            out_channels=num_channels,
            residual=True,
            use_3x3=res > 2
        )
        self.resnet.conv_d.weight.data *= np.sqrt(1 / n_blocks)
        
        self.proj_z = nn.Conv2d(self.latent_dim, num_channels, kernel_size=1, stride=1, padding=0)
        self.proj_z.weight.data *= np.sqrt(1 / res)
        
    def reparameterize(self, pm, pv):
        std = torch.exp(pv)
        eps = torch.randn_like(std)
        return pm + eps * std
    
    def forward(self, activation, params):
        if params.shape[0] != activation.shape[0]:
            params = params.repeat(activation.shape[0], 1, 1, 1)
        qm, qv = self.encoding_block(torch.cat([activation, params], dim=1)).chunk(2, dim=1)
        z = self.reparameterize(qm, qv)
        features = self.prior(params)
        pm, pv, prior = features[:, :self.latent_dim], features[:, self.latent_dim:2 * self.latent_dim], features[:, 2 * self.latent_dim:]
        kl_div = gaussian_analytical_kl(qm, qv, pm, pv) 
        x = self.resnet(params + prior + self.proj_z(z))
        return x, kl_div, z

    def uncond_forward(self, params, t=None, lvs=None):
        params = params[self.res]
        features = self.prior(params)
        pm, pv, prior = features[:, :self.latent_dim], features[:, self.latent_dim:2 * self.latent_dim], features[:, 2 * self.latent_dim:]
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = self.reparameterize(pm, pv)
        x = self.resnet(params + prior + self.proj_z(z))
        return x, z
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, dec_blocks, num_channels, bottleneck_ratio) -> None:
        super().__init__()
        blockstr = parse_layer_string(dec_blocks)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    latent_dim, 
                    res, 
                    num_channels, 
                    bottleneck_ratio, len(blockstr),
                    mixin=mixin
                ) for res, mixin in blockstr
            ]
        )
        
        self.resolutions = np.unique([res for res, _ in blockstr])
        self.activations_bias = nn.ParameterList(
            nn.Parameter(torch.zeros(1, num_channels, res, res)) for res in self.resolutions
        )
        self.gain = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.out = nn.Conv2d(num_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, activations):
        stats, xs = {}, {x.shape[2]: x for x in self.activations_bias}
        for block in self.decoder:
            x, kld, z = block(activations[block.res], xs[block.res])
            xs[block.res] = x
            stats[block.res] = dict(activation=x, kld=kld, z=z)
        output = stats[self.resolutions[-1]]['activation'] * self.gain + self.bias
        output = torch.sigmoid(self.out(output))
        return output, stats

    def uncond_forward(self, n_sample, t=None):
        xs = {}
        for bias in self.activations_bias:
            xs[bias.shape[2]] = bias.repeat(n_sample, 1, 1, 1)
        for block in self.decoder:
            xs, z = block.uncond_forward(xs, t=t)
        output = xs[self.resolutions[-1]] * self.gain + self.bias
        output = torch.sigmoid(self.out(output))
        return output, z

class VDVAE(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, num_channels, enc_blocks, dec_blocks, bottleneck_ratio) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, num_channels, enc_blocks, bottleneck_ratio)
        self.decoder = Decoder(latent_dim, out_channels, dec_blocks, num_channels, bottleneck_ratio)
        
    def forward(self, x):
        activations = self.encoder(x)
        output, stats = self.decoder(activations)
        return output, stats

    def sample(self, n_sample, t=None):
        sample, z = self.decoder.uncond_forward(n_sample, t=t)
        return sample, z
        