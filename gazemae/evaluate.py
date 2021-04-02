import logging
from datetime import datetime
from math import ceil

import pandas as pd
import numpy as np
from torch import no_grad, Tensor, manual_seed
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.dummy import DummyClassifier

from data import get_corpora
from data.data import SignalDataset
from network import ModelManager
from evals.classification_tasks import *
from evals.classifier_settings import *
from evals.utils import *
from settings import *


np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class RepresentationEvaluator:
    def __init__(self, tasks, classifiers='all', args=None, **kwargs):
        logging.info('\n---------- Initializing evaluator ----------')

        self.save_tsne_plot = args.save_tsne_plot
        self.scatterplot_dims = 2
        self.slice_time_windows = args.slice_time_windows
        self.viewing_time = kwargs.get('viewing_time') or args.viewing_time
        self.batch_size = args.batch_size if args.batch_size < 256 else 256
        self.tasks = tasks
        self.scorers = ['accuracy']  # f1_micro
        if classifiers == 'all':
            self.classifiers = list(CLASSIFIER_PARAMS.values())
        else:
            self.classifiers = [CLASSIFIER_PARAMS[c] for c in classifiers]

        # evaluate while training; use model passed in by trainer
        if 'model' in kwargs:
            self._caller = 'trainer'
            self.model = kwargs['model']
            self.representation_name = kwargs['representation_name']
            self.signal_types = [args.signal_type]
            self.feature_type_idxs = []
            self.dataset = {args.signal_type: kwargs['dataset']}
            # initialize own data set if the tranier is using sliced samples
            if not kwargs['dataset']:
                self.slice_time_windows = None
                self.dataset = self.init_dataset(args, **kwargs)

        # evaluate using pretrained models
        elif args.model_pos or args.model_vel:
            self._caller = 'main'
            self.model = ModelManager(args, training=False)
            self.representation_name = '{}-{}-pos-{}-vel-{}'.format(
                run_identifier,
                args.slice_time_windows,
                args.model_pos.split('/')[-1],
                args.model_vel.split('/')[-1])
            self.signal_types = list(self.model.network.keys())
            self.feature_type_idxs = self._build_representation_index_ranges()
            self.fi_df = pd.DataFrame()
            self.dataset = self.init_dataset(args)

        # evaluate using PCA features on raw data
        elif args.pca_components > 0:
            self.pca_components = args.pca_components
            self._caller = 'main'
            self.model = None
            self.representation_name = 'evaluate-pca-{}'.format(
                args.pca_components)
            self.signal_types = [args.signal_type]
            self.dataset = self.init_dataset(args)

        elif 'supervised' in kwargs['representation_name']:
            self.representation_name = kwargs['representation_name']
            self.signal_types = kwargs['signal_types']
            self.dataset = self.init_dataset(args)

        # will store each corpus' info into a unified self.df.
        # this will hold each trial's representation
        self.df = pd.DataFrame(columns=['corpus', 'subj', 'stim', 'task'])

        self.tensorboard = (SummaryWriter(
            'tensorboard_evals/{}'.format(self.representation_name))
            if args.tensorboard else None)

        self.consolidate_corpora()

    def consolidate_corpora(self):
        def get_corpus_df(corpus_name, signal_type):
            corpus = self.dataset[signal_type].corpora[corpus_name]

            if not self.slice_time_windows:
                corpus.data['corpus'] = corpus_name
                return corpus.data

            else:
                return corpus.load_slices_metadata()

        _signal = self.signal_types[0]
        for corpus_name in self.dataset[_signal].corpora.keys():
            corpus_df = get_corpus_df(corpus_name, self.signal_types[0])
            logging.info('{} {} signals loaded. Found {} trials'.format(
                corpus_name, self.signal_types[0], len(corpus_df)))

            if len(self.signal_types) > 1:
                corpus_df_2 = get_corpus_df(corpus_name, self.signal_types[1])
                logging.info('{} {} signals loaded. Found {} trials'.format(
                    corpus_name, self.signal_types[1], len(corpus_df_2)))

                # the original number of trials should be retained
                assert len(corpus_df) == len(corpus_df_2)
                col = 'in_{}'.format(self.signal_types[1])
                corpus_df[col] = corpus_df_2[col]

            self.df = pd.concat([self.df, corpus_df], sort=False)

        logging.info('Loaded corpora ({}) to Evaluator. {} total trials found'.format(
            self.signal_types, len(self.df)))

    def init_dataset(self, args, **kwargs):
        datasets = {}
        for signal_type in self.signal_types:
            args.signal_type = signal_type
            args.slice_time_windows = self.slice_time_windows

            if self._caller == 'trainer' and args.signal_type == 'vel':
                corpora = get_corpora(args, 'MIT-LowRes')
            else:
                corpora = get_corpora(args)

            datasets[signal_type] = SignalDataset(corpora, args,
                                                  # caller='evaluator',
                                                  load_to_memory=True,
                                                  **kwargs)
        return datasets

    def extract_representations(self, e=None, log_stats=False):
        def get_slice_representations(trial):
            corpus_hdf5 = dataset.corpora[trial.corpus].data
            trial_batch = np.stack([corpus_hdf5[i] for i in trial[in_col]])
            if dataset.normalize:
                trial_batch = np.stack([dataset.normalize_sample(s)
                                        for s in trial_batch])

            return np.mean(self.get_autoencoder_representations(
                           network, trial_batch), 0)

        if self.model is None and self.pca_components > 0:
            logging.info('\nExtracting PCA representations...')
            pca_features = pca(
                self.pca_components,
                self.dataset.samples.reshape(len(self.df), -1))
            self.df['z'] = list(pca_features)
            return

        z_cols = []
        for signal_type in self.signal_types:
            logging.info('\nExtracting {} representations...'.format(
                signal_type))
            dataset = self.dataset[signal_type]
            try:
                network = self.model.network[signal_type]
            except TypeError:
                network = self.model.network

            z_col = 'z_' + signal_type
            z_cols.append(z_col)
            in_col = 'in_' + signal_type

            if self.slice_time_windows:
                self.df[z_col] = self.df.apply(get_slice_representations, 1)

            else:
                _input = (np.stack(self.df[in_col]) if self.viewing_time > 0
                          else self.df[in_col])
                self.df[z_col] = self.get_autoencoder_representations(
                    network, _input)

        self.df['z'] = self.df.apply(
            lambda x: np.concatenate([x[col] for col in z_cols]),
            axis=1)
        logging.info('Done. Final representation shape: {}'.format(
            self.df['z'].iloc[0].shape))

        if log_stats:
            self._log_z_stats(e)

        # for analysis
        # tsne_plot_fifa(self.df)
        # tsne_plot_etra(self.df)
        # tsne_plot_corpus(self.df)

    def get_autoencoder_representations(self, network, x):
        if len(x.shape) > 2:
            batch_size = self.batch_size
            if x.shape[1] > x.shape[2]:
                x = x.swapaxes(1, 2)
        else:
            x = x.to_numpy()
            batch_size = 1

        reps = []
        network.eval()
        with no_grad():
            for s in range(ceil(len(x) / batch_size)):
                if batch_size == 1:
                    batch = Tensor(x[s]).T.unsqueeze(0)
                else:
                    batch = Tensor(x[batch_size * s: batch_size * (s + 1)])
                reps.extend(network.encode(batch.cuda()
                                           )[0].cpu().detach().numpy())
        return reps

    def evaluate(self, e=None):
        scores = {}
        for i, task in enumerate(self.tasks):
            _task = task.__class__.__name__  # convenience var
            logging.info('\nTask {}: {}'.format(i + 1, _task))
            x, y, = task.get_xy(self.df)
            if len(x) < 1:
                continue

            n_fold, refit, test_set = 5, False, None
            if _task == 'Biometrics_EMVIC':  # to compare with LPiTrack
                n_fold, refit, test_set  = 4, 'accuracy', task.get_test(self.df)

            self._log_labels(x, y)
            self._write_scatterplot(_task, x, y, e)
            # self._run_dummy_classifier(x, y)

            scores[task.__class__.__name__] = {}
            for classifier, params_grid in self.classifiers:
                if self._caller != 'trainer' and classifier[0] == 'svm_linear':
                    refit = 'accuracy'  # for feature importances

                pipeline = Pipeline([('scaler', StandardScaler()), classifier])

                grid_cv = GridSearchCV(pipeline, params_grid, cv=n_fold,
                                       n_jobs=4,
                                       scoring=self.scorers,
                                       refit=refit)
                grid_cv.fit(np.stack(x), y)

                acc = grid_cv.cv_results_['mean_test_accuracy'].max()
                logging.info('[{}] Acc: {:.4f}'.format(classifier[0], acc))

                scores[_task][classifier[0]] = acc

                if test_set is not None:
                    x_, y_ = test_set
                    # self._log_labels(x_, y_)
                    test_acc = grid_cv.score(np.stack(x_), y_)
                    logging.info('Test Acc: {:.4f}'.format(test_acc))
                    _task += '_test'
                    scores[_task] = {}
                    scores[_task][classifier[0]] = test_acc
                    self._write_scatterplot(_task, x_, y_, e)

                try:
                    logging.info('Best params: {}'.format(grid_cv.best_params_))
                    self._log_feature_importances(_task, grid_cv)
                except AttributeError:
                    pass

        self._write_fi_plots()
        return scores

    def _log_labels(self, x, y):
        if self._caller == 'main':
            labels = np.unique(y, return_counts=True)
            logging.info(
                '{} samples, {} Classes: '.format(len(x), len(labels[0])))
            logging.info(
                'Class Counts: {}'.format(dict(zip(*map(list, labels)))))

    def _write_fi_plots(self):
        if len(self.feature_type_idxs) < 2:
            # if not self.tensorboard or len(self.feature_type_idxs) < 2:
            return

        classifier = 'svm_linear'
        plots = plot_feature_importance(self.fi_df, self.signal_types,
                                        classifier)
        if self.tensorboard:
            for k, plot in plots.items():
                self.tensorboard.add_figure('top_features_{}'.format(k), plot)
                logging.info('[top features] {} plot saved to tensorboard'.format(
                    k))

    def _write_scatterplot(self, task, x, y, e):
        def add_figure(df, method, title_suffix=''):
            title = '_'.join([self.representation_name, task, method])

            if e is not None:  # means training, dont save each plot to disk
                savefig_title = None
            else:
                savefig_title = title + title_suffix
            fig = plot_scatter(df, savefig_title)

            if not self.tensorboard:
                return
            self.tensorboard.add_figure(title, fig, global_step=e)
            logging.info('{} scatterplot saved to tensorboard'.format(method))

        if self.save_tsne_plot:
            z_values = StandardScaler().fit_transform(np.stack(x))
            df = pd.DataFrame(TSNE(self.scatterplot_dims,
                                   perplexity=30,
                                   learning_rate=500,
                                   n_jobs=3).fit_transform(z_values))
            df['label'] = list(y)
            add_figure(df, 'tSNE', '-p{}-lr{}'.format(30, 500))

    def _log_feature_importances(self, task, grid_cv, top_n_percent=0.2):
        if len(self.feature_type_idxs) < 2:
            return

        _cls, classifier = grid_cv.best_estimator_.steps[1]
        num_top_features = int(self.total_latent_size * top_n_percent)
        if 'coef_' in dir(classifier):
            # coef_ is of shape (num_samples, latent_size), but just take ave.
            f = classifier.coef_.mean(0)

        elif 'feature_importances_' in dir(classifier):  # tree classifiers
            f = classifier.feature_importances_

        top_values_idx = f.argsort()[-num_top_features:]
        top_values = f[top_values_idx]
        for feature_type, (start, end) in self.feature_type_idxs.items():
            in_idx_range = (top_values_idx >= start) & \
                           (top_values_idx < end)
            logging.info('\t[{} features in top {}%] Count: {}'.format(
                feature_type, int(top_n_percent * 100), in_idx_range.sum()))

            values = top_values[in_idx_range]
            if len(values) > 0:
                logging.info('\t\tAve. coef value: {:.2f} (min: {:.2f}, max: {:.2f})'.format(
                    values.mean(), values.min(), values.max()))

            self.fi_df = self.fi_df.append(
                pd.DataFrame({'task': task,
                              'classifier': _cls,
                              feature_type: values}),
                sort=False)

    def _build_representation_index_ranges(self):
        idx = 0
        signal_idxs = {}
        for signal_type in self.signal_types:
            _network = self.model.network[signal_type]
            _latent_size = _network.latent_size

            if _network.hierarchical:
                signal_idxs[signal_type + '-2'] = (idx, idx + _latent_size)
                idx += _latent_size
                signal_idxs[signal_type + '-1'] = (idx, idx + _latent_size)
                idx += _latent_size
            else:
                signal_idxs[signal_type] = (idx, idx + _latent_size)
                idx += _latent_size

        logging.info('\nSignal types and their indeces in Z:\n{}'.format(signal_idxs))
        self.total_latent_size = idx
        return signal_idxs

    def _run_dummy_classifier(self, x, y):
        dummy_pipeline = Pipeline([
            ('scaler', StandardScaler()), ('dummy', DummyClassifier())])
        dummy_scores = cross_validate(
            dummy_pipeline, np.stack(x), y, cv=5, scoring=self.scorers)
        logging.info('Chance Mean Acc: {:.2f}'.format(
            np.mean(dummy_scores['test_accuracy'])))

    def _log_z_stats(self, e):
        def log(stat, values):
            logging.info('[{} of latent space dims] Total mean: {:.2f}. Values:\n{}'.format(
                stat, values.mean(), values))

        z = np.stack(self.df['z'])
        z_means = z.mean(0)  # mean of each dimension
        z_std = z.std(0)  # std of each dimension

        if self.tensorboard:
            self.tensorboard.add_figure(
                'Z_Means', plot_hist(z_means, 'Means per Z dimension'),
                global_step=e)
            self.tensorboard.add_scalar('Z_Mean_of_Means', z_means.mean(), e)
            self.tensorboard.add_figure(
                'Z_StDevs', plot_hist(z_std, 'Std. Devs per Z dimension'),
                global_step=e)
            self.tensorboard.add_scalar('Z_Mean_of_StDevs', z_std.mean(), e)
            logging.info('Stat of Z dimensions added to Tensorboard.')


if __name__ == '__main__':
    run_identifier = 'eval_' + datetime.now().strftime('%m%d-%H%M')
    args = get_parser().parse_args()
    setup_logging(args, run_identifier)
    evaluator = RepresentationEvaluator(tasks=[
        Biometrics_EMVIC(),
        Biometrics(),
        # Biometrics_MIT_LR(),
        ETRAStimuli(),
        AgeGroupBinary(),
        GenderBinary(),
    ], args=args)
    evaluator.extract_representations()
    evaluator.evaluate()
