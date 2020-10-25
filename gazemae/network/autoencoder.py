import logging

from torch import nn, manual_seed, exp, randn_like, randn

from network.encoder import Encoder
from network.decoder import *

from settings import *


manual_seed(RAND_SEED)


class TCNAutoencoder(nn.Module):
    def __init__(self, args):
        super(TCNAutoencoder, self).__init__()
        self.is_vae = 'vae' in args.loss_type
        self.latent_size = LATENT_SIZE
        self.hierarchical = args.hierarchical
        if self.hierarchical:
            self.latent_size = int(self.latent_size / 2)

        self.encoder = Encoder(args, *ENCODER_PARAMS)

        self.decoder = CausalDecoder(
            args, *DECODER_PARAMS, self.latent_size)

        self.bottleneck_fns = nn.ModuleDict(
            {'1': self.get_bottleneck_fns(self.encoder.out_dim)})
        if self.hierarchical:
            self.bottleneck_fns.update(
                {'2': self.get_bottleneck_fns(
                    self.encoder.out_dim2)})

    def get_bottleneck_fns(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_size))

    def forward(self, x, is_training=False):
        z, mean, logvar = self.encode(x, cat_output=False)
        out = self.decoder(z, x, is_training)

        return out, z, mean, logvar

    def bottleneck(self, x, level='1', prev_z=None):
        # for when batch size = 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)

        fc_fn = self.bottleneck_fns[level]
        if prev_z is not None:
            x = cat((x, prev_z), -1)
        z = fc_fn(x)

        logging.debug('Z: {}'.format(x.shape))
        return z, None, None

    def encode(self, x, cat_output=True):
        if self.hierarchical:
            enc_out1, enc_out2 = self.encoder(x)
            z1, mean1, logvar1 = self.bottleneck(enc_out1, level='1')
            z2, mean2, logvar2 = self.bottleneck(enc_out2, level='2')

            if not cat_output:
                return [z2, z1], [mean2, mean1], [logvar2, logvar1]

            z = cat([z2, z1], -1)
            mean, logvar = None, None
            return z, mean, logvar

        else:
            return self.bottleneck(self.encoder(x))
