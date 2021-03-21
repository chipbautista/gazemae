
import time
import logging
from datetime import datetime

import numpy as np
from torch import manual_seed, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_corpora
from data.data import SignalDataset
from network import ModelManager
from evaluate import *
from evals.classification_tasks import *
from evals.utils import *
from settings import *

np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class Trainer:
    def __init__(self):
        self.model = ModelManager(args)

        self.save_model = args.save_model
        self.cuda = args.cuda
        if self.cuda:
            self.model.network = self.model.network.cuda()

        self.rec_loss = args.rec_loss

        self._load_data()
        self._init_loss_fn(args)
        self._init_evaluator()

    def _load_data(self):
        self.dataset = SignalDataset(get_corpora(args), args,
                                     caller='trainer')

        _loader_params = {'batch_size': args.batch_size, 'shuffle': True,
                          'pin_memory': True}

        if len(self.dataset) % args.batch_size == 1:
            _loader_params.update({'drop_last': True})

        self.dataloader = DataLoader(self.dataset, **_loader_params)
        self.val_dataloader = (
            DataLoader(self.dataset.val_set, **_loader_params)
            if self.dataset.val_set else None)

    def _init_loss_fn(self, args):
        self._loss_types = ['total']

        # just to keep track of all the losses i have to log
        self._loss_types.append('rec')
        self.loss_fn = nn.MSELoss(reduction='none')

    def _init_evaluator(self):
        # for logging out this run
        _rep_name = '{}{}-hz:{}-s:{}'.format(
            run_identifier, 'mse',
            self.dataset.hz, self.dataset.signal_type)

        self.evaluator = RepresentationEvaluator(
            tasks=[Biometrics_EMVIC(), ETRAStimuli(),
                   AgeGroupBinary(), GenderBinary()],
            # classifiers='all',
            classifiers=['svm_linear'],
            args=args, model=self.model,
            # the evaluator should initialize its own dataset if the trainer
            # is using manipulated trials (sliced, transformed, etc.)
            dataset=(self.dataset if not args.slice_time_windows
                     else None),
            representation_name=_rep_name,
            # to evaluate on whole viewing time
            viewing_time=-1)

        if args.tensorboard:
            self.tensorboard = SummaryWriter(
                'tensorboard_runs/{}'.format(_rep_name))
        else:
            self.tensorboard = None

    def reset_epoch_losses(self):
        self.epoch_losses = {'train': {l: 0.0 for l in self._loss_types},
                             'val': {l: 0.0 for l in self._loss_types}}

    def init_global_losses(self, num_checkpoints):
        self.global_losses = {
            'train': {l: np.zeros(num_checkpoints) for l in self._loss_types},
            'val': {l: np.zeros(num_checkpoints) for l in self._loss_types}}

    def update_global_losses(self, checkpoint):
        for dset in ['train', 'val']:
            for l in self._loss_types:
                self.global_losses[dset][l][checkpoint] = self.epoch_losses[dset][l]

    def train(self):
        logging.info('\n===== STARTING TRAINING =====')
        logging.info('{} samples, {} batches.'.format(
                     len(self.dataset), len(self.dataloader)))
        logging.info('Loss Fn:' + str(self.loss_fn))

        _checkpoint_interval = len(self.dataloader)
        num_checkpoints = int(MAX_TRAIN_ITERS / _checkpoint_interval)
        self.init_global_losses(num_checkpoints + 1)

        i, e = 0, 0
        _checkpoint_start = time.time()
        while i < MAX_TRAIN_ITERS:
            self.reset_epoch_losses()

            for b, batch in enumerate(self.dataloader):
                self.model.network.train()
                sample, sample_rec = self.forward(batch)

                i += 1
                if i % _checkpoint_interval == 0:
                    self.update_global_losses(int(i / _checkpoint_interval) - 1)
                    self.log(i, _checkpoint_interval,
                             time.time() - _checkpoint_start)
                    self.reset_epoch_losses()

                    if (e + 1) % 10 == 0:
                        self.evaluate_representation(sample, sample_rec, i)
                        if self.save_model:
                            self.model.save(i, run_identifier, self.global_losses)

                    _checkpoint_start = time.time()

            e += 1

    def forward(self, batch):
        batch = batch.float()
        if self.cuda:
            batch = batch.cuda()

        _is_training = self.model.network.training
        out = self.model.network(batch, is_training=_is_training)

        dset = 'train' if self.model.network.training else 'val'

        rec_batch = out[0]
        loss = self.loss_fn(rec_batch, batch
                            ).reshape(rec_batch.shape[0], -1).sum(-1).mean()
        self.epoch_losses[dset]['total'] += loss.item()

        if self.model.network.training:
            loss.backward()
            self.model.optim.step()
            self.model.optim.zero_grad()

        rand_idx = np.random.randint(0, batch.shape[0])
        return batch[rand_idx].cpu(), rec_batch[rand_idx].cpu()

    def evaluate_representation(self, sample, sample_rec, i):
        if sample is not None:
            viz = visualize_reconstruction(
                sample, sample_rec,
                filename='{}-{}'.format(run_identifier, i),
                loss_func=self.rec_loss,
                title='[{}] [i={}] vl={:.2f} vrl={:.2f}'.format(
                    self.rec_loss, i, self.epoch_losses['val']['total'],
                    self.epoch_losses['val']['rec']),
                savefig=False if self.tensorboard else True)

            if self.tensorboard:
                self.tensorboard.add_figure('e_{}'.format(i),
                                            figure=viz,
                                            global_step=i)

        self.evaluator.extract_representations(i, log_stats=True)
        scores = self.evaluator.evaluate(i)
        if self.tensorboard:
            for task, classifiers in scores.items():
                for classifier, acc in classifiers.items():
                    self.tensorboard.add_scalar(
                        '{}_{}_acc'.format(task, classifier), acc, i)

    def log(self, i, num_train_iters, t):
        def get_mean_losses(dset):
            try:
                iters = (num_train_iters if dset == 'train'
                         else len(self.val_dataloader))
            except TypeError:
                iters = 1
            return {loss: self.epoch_losses[dset][loss] / iters
                    for loss in self._loss_types}

        def stringify(losses):
            return ' '.join(['{}: {:.2f}'.format(loss.upper(), val)
                             for (loss, val) in losses.items()
                             if loss != 'total'])

        def to_tensorboard(dset, losses):
            for (loss, val) in losses.items():
                self.tensorboard.add_scalar(
                    '{}_{}_loss'.format(dset, loss), val, i)

        tr_losses = get_mean_losses('train')
        val_losses = get_mean_losses('val')

        # build string to print out
        string = '[{}/{}] TLoss: {:.4f}, VLoss: {:.4f} ({:.2f}s)'.format(
            i, MAX_TRAIN_ITERS, tr_losses['total'], val_losses['total'], t)
        string += '\n\t train ' + stringify(tr_losses)
        if val_losses['total'] > 0.00:
            string += '\n\t val ' + stringify(val_losses)
        logging.info(string)

        if self.tensorboard:
            to_tensorboard('train', tr_losses)
            if val_losses['total'] > 0.00:
                to_tensorboard('val', val_losses)


args = get_parser().parse_args()
run_identifier = datetime.now().strftime('%m%d-%H%M')
setup_logging(args, run_identifier)
print_settings()

logging.info('\nRUN: ' + run_identifier + '\n')
logging.info(str(args))

trainer = Trainer()
trainer.train()
