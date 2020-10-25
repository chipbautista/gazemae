import logging
from torch import nn, cat


class Encoder(nn.Module):
    def __init__(self, args, kernel_size, filters, dilations, downsamples,
                 in_channels=2):
        super(Encoder, self).__init__()
        self.hierarchical = args.hierarchical
        self.multiscale = args.multiscale
        self.causal = args.causal_encoder
        self.in_dim = int(args.hz * args.viewing_time)
        self.in_channels = in_channels

        self.kernel_size = kernel_size
        self.filters = filters
        self.dilations = dilations
        self.downsamples = downsamples

        if self.hierarchical:
            self.out_dim = self.filters[1]
            self.out_dim2 = self.filters[3]
        elif self.multiscale:
            self.out_dim = sum([f for f in self.filters[1::2]])
            # self.out_dim = sum(self.filters)
        else:
            self.out_dim = self.filters[-1]

        self.blocks = []
        for block_num, (f, dil, down) in enumerate(zip(self.filters,
                                                       self.dilations,
                                                       self.downsamples)):

            self.blocks.append(
                ResidualBlock(
                    in_channels=(self.in_channels if block_num == 0
                                 else self.filters[block_num - 1]),
                    mid_channels=f,
                    out_channels=f,
                    dilations=dil,
                    kernel_size=self.kernel_size,
                    causal=self.causal,
                    downsample=down
                )
            )

        self.blocks = nn.Sequential(*self.blocks)
        self.out_dim = int(self.out_dim)
        logging.info('\nEncoder initialized.')
        logging.info('Multiscale Rep.: {}'.format(self.multiscale))
        logging.info('Global Pool: {}'.format('True -- PERMA DEFAULT'))
        logging.info('Causal: {}'.format(self.causal))
        logging.info('Downsample amounts per block: {}\n'.format(self.downsamples))

    def forward(self, x):
        if self.hierarchical:
            out_1 = self.blocks[1](self.blocks[0](x))
            out_2 = self.blocks[3](self.blocks[2](out_1))
            return (out_1.mean(-1), out_2.mean(-1))

        if not self.multiscale:
            x = self.blocks(x)
            return x.mean(-1)  # Global Average Pooling

        block_features = []
        for block_num, block in enumerate(self.blocks):
            x = block(x)
            if block_num % 2 == 1:
                block_features.append(x.mean(-1).squeeze())
        return cat(block_features, -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dilations,
                 kernel_size, causal, downsample=0, no_skip=False, no_bn2=False):
        super(ResidualBlock, self).__init__()
        self.causal = causal
        self.kernel_size = kernel_size

        self.relu = nn.ReLU()
        self.conv1 = self._build_conv_layer(
            in_channels, mid_channels, dilations[0])
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = self._build_conv_layer(
            mid_channels, out_channels, dilations[1])
        self.bn2 = (nn.BatchNorm1d(out_channels) if not no_bn2 else None)

        self.skip_conv = (nn.Conv1d(in_channels, out_channels, 1)
                          if not no_skip else None)

        if downsample > 0:
            self.downsample = nn.MaxPool1d(downsample, downsample)
        else:
            self.downsample = None

    def _build_conv_layer(self, in_ch, out_ch, dilation):
        if dilation >= 1:
            padding = dilation * (self.kernel_size - 1)
            if self.causal:
                pad_sides = (padding, 0)
            else:
                pad_sides = int(padding / 2)
            return nn.Sequential(
                nn.ConstantPad1d(pad_sides, 0),
                nn.Conv1d(in_ch, out_ch, self.kernel_size, dilation=dilation))
        else:
            return nn.Conv1d(in_ch, out_ch, self.kernel_size)

    def forward(self, x):
        out = self.bn1(self.relu(self.conv1(x)))

        out = self.conv2(out)

        if self.skip_conv is not None:
            out = out + self.skip_conv(x)
        out = self.relu(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            out = self.downsample(out)
        return out


class WaveNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, kernel_size, causal, downsample=0, no_skip=False, no_bn2=False):
        super(WaveNetBlock, self).__init__()
        self.causal = causal
        self.kernel_size = kernel_size

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)



        self.skip_conv = (nn.Conv1d(in_channels, out_channels, 1)
                          if not no_skip else None)

        if downsample > 0:
            self.downsample = nn.MaxPool1d(downsample, downsample)
        else:
            self.downsample = None

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = self.tanh(conv_out) * self.sigmoid(conv_out)
        residual_out = conv_out + x
        return conv_out, residual_out
