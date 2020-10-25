
from torch import nn, manual_seed

from network.encoder import Encoder

from settings import *


manual_seed(RAND_SEED)


class SupervisedTCN(nn.Module):
    def __init__(self, args, num_classes, in_channels):
        super(SupervisedTCN, self).__init__()
        self.encoder = Encoder(args, *ENCODER_PARAMS, in_channels=in_channels)
        self.logits = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x, is_training=False):
        return self.logits(self.encoder(x))
