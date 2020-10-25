import logging

from numpy import random, arange
from torch import nn, cat, zeros

from .encoder import ResidualBlock
from settings import *

random.seed(RAND_SEED)


class CausalDecoder(nn.Module):
    def __init__(self, args, kernel_size, filters, dilations, input_dropout,
                 latent_dim):
        super(CausalDecoder, self).__init__()
        self.in_dim = int(args.hz * args.viewing_time)
        self.in_channels = 2

        self.input_dropout = input_dropout  # prob to zero
        self.filters = filters
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.blocks = nn.ModuleList([])
        self.latent_projections = nn.ModuleList([])

        for block_num, f in enumerate(self.filters):
            if block_num == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.filters[block_num - 1]

            self.blocks.append(
                CausalBlock(
                    in_channels=in_channels,
                    mid_channels=f,
                    out_channels=(f if block_num != len(self.filters) - 1
                                  else self.in_channels),
                    dilations=self.dilations[block_num],
                    kernel_size=self.kernel_size,
                    causal=True,
                    no_bn2=block_num == len(self.filters) - 1)
            )

            if block_num % 2 == 0:
                self.latent_projections.append(
                    nn.Linear(latent_dim, f))
            else:
                self.latent_projections.append(None)

        logging.info('Dilated Causal Decoder initialized.')
        logging.info('Input dropout p (to zero): {}'.format(self.input_dropout))

    def forward(self, z, x_true, is_training):
        # pad the input at its left so there is no leak from input t=1 to
        # output t=1. should be: output for t=1 is dependent on input t=0
        x = cat((x_true, zeros(x_true.shape[0], 2, 1).cuda()), dim=2)

        x = nn.functional.dropout(x, self.input_dropout, is_training)
        """
        IMPORTANT! Dropout used in NNs don't actually just drop out values
        It scales non-dropped values by 1/p so that the same graph is used in
        backend, I think. So this adds noise to this "interpolative" decoder.
        It's a wrong implementation on my end, but doesn't void the results.
        """

        if isinstance(z, list):  # 2 latent spaces
            projections = [p(z[int(i / 2)]) if p else None
                           for (i, p) in enumerate(self.latent_projections)]
        else:
            projections = [p(z) if p else None for p in self.latent_projections]

        for block, latent_proj in zip(self.blocks, projections):
            if latent_proj is not None:
                x = block(x, latent_proj)
            else:
                x = block(x)
        return x[:, :, :-1]


class CausalBlock(ResidualBlock):
    def forward(self, x, projection=None):
        out = self.conv1(x)
        if projection is not None:
            out = out + projection.unsqueeze(-1)
        out = self.bn1(self.relu(out))

        out = self.conv2(out)
        if self.skip_conv is not None:
            out = out + self.skip_conv(x)
        out = self.relu(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        return out


class AutoregressiveDecoder(nn.Module):
    def __init__(self, args, kernel_size, filters, dilations, latent_dim):
        super(AutoregressiveDecoder, self).__init__()
        self.p_teacher_forcing = args.p_teacher_forcing
        self.in_dim = int(args.hz * args.viewing_time)
        self.in_channels = 2
        self.kernel_size = kernel_size
        self.filters = filters
        self.dilations = dilations

        self.blocks = nn.ModuleList([])
        self.latent_projections = nn.ModuleList([])
        for block_num, (f, dilations) in enumerate(zip(self.filters,
                                                       self.dilations)):
            if block_num == 0:
                in_ch = self.in_channels * self.kernel_size
                in_queue_dim = self.in_channels

            else:
                in_ch = self.filters[block_num - 1] * self.kernel_size
                in_queue_dim = self.filters[block_num - 1]

            # in_ch += latent_dim
            self.blocks.append(
                AutoregressiveResidualBlock(
                    in_ch, f,
                    out_ch=(f if block_num != len(self.filters) - 1
                            else self.in_channels),
                    kernel_size=self.kernel_size,
                    dilations=dilations,
                    # override queue dimension for the first layer bec
                    # of concatenation with latent Z
                    in_queue_dim=in_queue_dim,
                    # do not perform residual connection on the last block?
                    is_last_block=(block_num + 1 == len(self.filters))
                )
            )
            if block_num % 2 == 0:
                self.latent_projections.append(nn.Linear(latent_dim, f))
            else:
                self.latent_projections.append(None)

        logging.info('Decoder initialized.')

    def _init_conv_queues(self, batch_size):
        try:
            for decoder in self.blocks:
                decoder.init_conv_queues(batch_size)
        except AttributeError:
            pass

    def decrement_teacher_forcing_p(self):
        if self.p_teacher_forcing > 0.0009:
            self.p_teacher_forcing -= 0.05
        logging.info('teacher forcing probability = {:.2f}'.format(
            self.p_teacher_forcing))

    def forward(self, z, x_true, is_training):
        def _do_teacher_force(i):
            return (self.p_teacher_forcing and i > 0 and is_training and
                    random.choice([True, False],
                                  p=[self.p_teacher_forcing,
                                     1 - self.p_teacher_forcing]))

        self._init_conv_queues(z.shape[0])
        predictions = zeros(z.shape[0], self.in_channels, self.in_dim).cuda()
        x = zeros(z.shape[0], 2, 1).cuda()

        # z_projection = self.latent_projections[0](z)
        z_projections = [p(z) if p else None for p in self.latent_projections]
        for i in range(self.in_dim + 1):

            for b, decoder in enumerate(self.blocks):
                if b == 0:
                    if _do_teacher_force(i):
                        x = x_true[:, :, i - 1].unsqueeze(-1)

                if z_projections[b] is not None:
                    x = decoder(x, z_projections[b])
                else:
                    x = decoder(x)

            if i > 0:
                predictions[:, :, i - 1] = x.squeeze()
        return predictions


class AutoregressiveResidualBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, dilations, kernel_size,
                 in_queue_dim=None, is_last_block=False):
        super(AutoregressiveResidualBlock, self).__init__()

        self.kernel_size = kernel_size
        self.dilation1, self.dilation2 = dilations
        self.relu = nn.ReLU()
        self.conv1 = nn.Linear(in_ch, mid_ch)
        self.bn1 = nn.BatchNorm1d(mid_ch)

        self.conv2 = nn.Linear(mid_ch * self.kernel_size, out_ch)

        if not is_last_block:
            if in_ch != out_ch:
                self.skip_conv = nn.Linear(in_ch, out_ch)
            else:
                self.skip_conv = None
            self.bn2 = nn.BatchNorm1d(out_ch)
        else:
            self.skip_conv = None
            self.bn2 = None

        logging.info('AutoregressiveResidualBlock initialized.' +
                     'Dilations: {} {}'.format(self.dilation1, self.dilation2))

        # for initializing convolution queues
        self._mid_queue_dim = mid_ch
        self._in_queue_dim = in_queue_dim if in_queue_dim else in_ch

    def init_conv_queues(self, batch_size):
        # each queue for a layer has length = dilation * ksize-1
        self.conv1_queue = zeros(
            self.dilation1 * (self.kernel_size - 1),
            batch_size, self._in_queue_dim, 1).cuda()
        self.conv2_queue = zeros(
            self.dilation2 * (self.kernel_size - 1),
            batch_size, self._mid_queue_dim, 1).cuda()

    def forward(self, x, z=None):
        def _get_dilated_nodes(queue, dilation, new_node):
            return cat((*[queue[i] for i in arange(0, len(queue), dilation)],
                        new_node), axis=2)

        def _left_shift_queue(queue, new_node):
            queue[:-1] = queue[1:]
            queue[-1] = new_node
            return queue

        bs = x.shape[0]

        # LAYER 1
        l1_input = _get_dilated_nodes(self.conv1_queue,
                                      self.dilation1, x).reshape(bs, -1)

        conv1_out = self.conv1(l1_input)

        # use projected Z as a bias to conv output
        if z is not None:
            conv1_out = conv1_out + z

        conv1_out = self.bn1(self.relu(conv1_out))

        # LAYER 2
        l2_input = _get_dilated_nodes(self.conv2_queue, self.dilation2,
                                      conv1_out.unsqueeze(-1)).reshape(bs, -1)

        conv2_out = self.conv2(l2_input)

        if self.skip_conv is not None:
            conv2_out = conv2_out + self.skip_conv(l1_input)
        conv2_out = self.relu(conv2_out)

        if self.bn2:
            conv2_out = self.bn2(conv2_out)

        # Push to queue
        self.conv1_queue = _left_shift_queue(self.conv1_queue, x)
        self.conv2_queue = _left_shift_queue(self.conv2_queue, conv1_out.unsqueeze(-1))

        return conv2_out.unsqueeze(-1)


class WaveNetBlock(nn.Module):
    def forward(self, x, z):
        x_filter = self.x_conv_filter(x)
        x_gate = self.x_conv_gate(x)
        z_filter = self.z_conv_filter(z)
        z_gate = self.z_conv_gate(z)

        out = self.tanh(x_filter + z_filter) * self.sigmoid(x_gate + z_gate)
        return self.bn(out)
