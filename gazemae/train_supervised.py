import logging
from datetime import datetime

from torch import no_grad
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import numpy as np

from network.supervised import SupervisedTCN
from evaluate import RepresentationEvaluator
from evals.classification_tasks import *
from settings import *


class SupervisedTrainer:
    def __init__(self, args):
        self.args = args
        assert args.task in TASKS.keys()
        self.task = TASKS[args.task]()

        signal_types = args.signal_type.split(',')

        # Use the evaluator to initialize the dataset
        self.evaluator = RepresentationEvaluator(
            tasks=None, args=args,
            representation_name=run_identifier,
            signal_types=signal_types)

        if len(signal_types) > 1:
            # concatenate pos and vel signals
            self.evaluator.df['z'] = self.evaluator.df.apply(
                lambda x: np.concatenate([x['in_pos'], x['in_vel']], 1), 1)
            self.in_channels = 4
        else:
            self.evaluator.df['z'] = self.evaluator.df['in_' + args.signal_type]
            self.in_channels = 2

        self.x, self.y = self.task.get_xy(self.evaluator.df)
        self.x = np.stack(self.x)
        self.y = self.y.to_numpy()

        self.num_samples = len(self.x)
        self.num_classes = len(np.unique(self.y))
        self.targets = LabelBinarizer().fit_transform(self.y)
        if self.num_classes == 2:
            self.targets = np.hstack((self.targets, 1 - self.targets))

        self.bce_loss = BCEWithLogitsLoss()

        logging.info('Supervised CNN initialized.')
        logging.info('Batch size: {}'.format(self.args.batch_size))
        logging.info('Learning rate: {}'.format(self.args.learning_rate))

    def init_network(self):
        network = SupervisedTCN(self.args, self.num_classes,
                                in_channels=self.in_channels)
        optim = Adam(network.parameters(), lr=self.args.learning_rate)
        logging.info('New CNN initialized.')
        return network.cuda(), optim

    def init_dataloaders(self, train_x, train_y, test_x, test_y):
        _params = {'batch_size': self.args.batch_size, 'shuffle': True}
        train_dataloader = DataLoader(SupervisedDataset(train_x, train_y),
                                      **_params)
        test_dataloader = DataLoader(SupervisedDataset(test_x, test_y),
                                     **_params)
        return train_dataloader, test_dataloader

    def train(self):
        x, y = self.task.get_xy(self.evaluator.df)
        x = x.to_numpy()

        if self.args.task == 'biometrics-emvic-test':
            # train_x, train_y = self.task.get_xy(self.evaluator.df)
            test_x, test_y = self.task.get_test(self.evaluator.df)
            test_x = test_x.to_numpy()

            one_hot = LabelBinarizer()
            y = one_hot.fit_transform(y)
            test_y = one_hot.transform(test_y)

            test_acc = self.train_network(
                0, *self.init_dataloaders(x, y, test_x, test_y))
            logging.info('Test acc: {:.4f}'.format(test_acc))
        else:
            val_accs = self.cross_validate(x, y)
            logging.info('Mean test acc: {:.4f}'.format(np.mean(val_accs)))

    def cross_validate(self, x, y):
        n_splits = 4 if self.args.task == 'biometrics-emvic' else 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        val_accs = []
        targets = LabelBinarizer().fit_transform(y)
        for f, (train_idxs, test_idxs) in enumerate(skf.split(x, y)):
            train_x, test_x = x[train_idxs], x[test_idxs]
            train_y, test_y = targets[train_idxs], targets[test_idxs]
            # train_x, val_x, train_y, val_y = train_test_split(
            #     train_x, train_y, test_size=0.2, stratify=train_y)

            val_acc = self.train_network(
                f, *self.init_dataloaders(train_x, train_y, test_x, test_y))
            val_accs.append(val_acc)
        return val_accs

    def train_network(self, f, train_dataloader, val_dataloader):
        def _log(*vals):
            return ' [{}] loss: {:.2f}. acc: {:.2f}'.format(*vals)

        network, optim = self.init_network()
        if f == 0:
            logging.info('\n ' + str(network))
            logging.info('# of Parameters: ' +
                         str(sum(p.numel() for p in network.parameters()
                                 if p.requires_grad)))

        num_train_samples = len(train_dataloader.dataset)
        num_val_samples = len(val_dataloader.dataset)
        logging.info('Starting training for fold={}.'.format(f))
        logging.info('Num train samples: {}'.format(num_train_samples))
        logging.info('Num train batches: {}'.format(len(train_dataloader)))
        logging.info('Num val samples: {}'.format(num_val_samples))
        logging.info('Num val batches: {}'.format(len(val_dataloader)))

        min_val_loss = 999
        epochs_without_decrease = 0
        for e in range(100):
            _log_string = 'Epoch: {}'.format(e)

            _loss, num_correct = 0, 0
            network.train()
            for x, y in train_dataloader:
                x = x.cuda().float()
                out = network.forward(x)
                # import pdb; pdb.set_trace()
                loss = self.bce_loss(out, y.cuda().float())
                loss.backward()
                optim.step()
                optim.zero_grad()

                num_correct += int((out.cpu().argmax(1) == y.argmax(1)).sum())
                _loss += loss.item()
            _log_string += _log('train', _loss, num_correct / num_train_samples)

            _loss, num_correct = 0, 0
            network.eval()
            with no_grad():
                for x, y in val_dataloader:
                    x = x.cuda().float()
                    out = network.forward(x)
                    loss = self.bce_loss(out, y.cuda().float())
                    num_correct += int((out.cpu().argmax(1) == y.argmax(1)).sum())
                    _loss += loss.item()
            acc = num_correct / num_val_samples
            _log_string += _log('val', _loss, acc)

            if _loss <= min_val_loss:
                min_val_loss = _loss
                max_val_acc = acc
                epochs_without_decrease = 0
            else:
                epochs_without_decrease += 1
                if epochs_without_decrease == 5:
                    logging.info('Val loss. not decreasing. Stopping training.')
                    break

            logging.info(_log_string)
        return max_val_acc


class SupervisedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i].T, self.y[i]

    def __len__(self):
        return len(self.x)


run_identifier = 'supervised_' + datetime.now().strftime('%m%d-%H%M')
args = get_parser().parse_args()
setup_logging(args, run_identifier)
trainer = SupervisedTrainer(args)
trainer.train()
