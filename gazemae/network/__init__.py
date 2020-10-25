import logging

import torch

from .autoencoder import TCNAutoencoder
from .supervised import SupervisedTCN


class ModelManager:
    def __init__(self, args, training=True, **kwargs):
        self.is_training = training
        self.load_network(args, **kwargs)

    def load_network(self, args, **kwargs):
        if args.model_pos or args.model_vel:
            if self.is_training:
                self.network, self.optim = self._load_pretrained_model(
                    args.model_pos or args.model_vel)
            else:
                self.network = {}
                vel_net = self._load_pretrained_model(args.model_vel)
                if vel_net:
                    self.network['vel'] = vel_net.eval()
                pos_net = self._load_pretrained_model(args.model_pos)
                if pos_net:
                    self.network['pos'] = pos_net.eval()

        else:
            if args.loss_type == 'supervised':
                self.network = SupervisedTCN(args, kwargs['num_classes'])
            else:
                self.network = TCNAutoencoder(args)

            self.optim = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)
            self._log(self.network)

    def _load_pretrained_model(self, model_name):
        if not model_name:
            return None

        logging.info('Loading saved model {}...'.format(model_name))
        model = torch.load('../models/' + model_name)
        try:
            network = model['network']
        except KeyError:
            network = model['model']
        self._log(network)

        network.load_state_dict(model['model_state_dict'])

        if self.is_training:
            optim = torch.optim.Adam(network.parameters())
            optim.load_state_dict(model['optimizer_state_dict'])
            return network, optim

        return network

    def _log(self, network):
        logging.info('\n ' + str(network))
        logging.info('# of Parameters: ' +
                     str(sum(p.numel() for p in network.parameters()
                             if p.requires_grad)))

    def save(self, i, run_identifier, losses):
        model_filename = '../models/' + run_identifier + '-i' + str(i)
        torch.save(
            {
                'iter': i,
                'network': self.network,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'losses': losses
            }, model_filename)
        logging.info('Model saved to {}'.format(model_filename))
