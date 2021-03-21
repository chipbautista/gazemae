import logging
from argparse import ArgumentParser

# training settings
MAX_TRAIN_ITERS = 20000
CHECKPOINT_INTERVAL = 500

# network settings
LATENT_SIZE = 128
ENCODER_FILTERS = [256, 256, 256, 256]
ENCODER_DILATIONS = [(1, 1), (2, 4), (8, 16), (32, 64)]
ENCODER_DOWNSAMPLE = [0, 0, 0, 0]
ENCODER_KERNEL_SIZE = 3

ENCODER_PARAMS = [ENCODER_KERNEL_SIZE, ENCODER_FILTERS, ENCODER_DILATIONS,
                  ENCODER_DOWNSAMPLE]

DECODER_FILTERS = [128, 128, 128, 128]
DECODER_DILATIONS = [(1, 1), (2, 4), (8, 16), (32, 64)]
DECODER_KERNEL_SIZE = 3
DECODER_INPUT_DROPOUT = 0.66  # BEST = 0.66 for vel, 0.75 for pos
DECODER_PARAMS = [DECODER_KERNEL_SIZE, DECODER_FILTERS, DECODER_DILATIONS,
                  DECODER_INPUT_DROPOUT]

# data settings
MAX_X_RESOLUTION = 1280
MAX_Y_RESOLUTION = 1024
DATA_ROOT = '../data/'
GENERATED_DATA_ROOT = '../generated-data/'

SLICE_OVERLAP_RATIO = 0.2

PX_PER_DVA = 35  # pixels per degree of visual angle
RAND_SEED = 123


def print_settings():
    logging.info({
        k: v for (k, v) in globals().items()
        if not k.startswith('_') and k.isupper()})


def setup_logging(args, run_identifier=''):
    logging.getLogger().setLevel(logging.INFO)
    if args.log_to_file:
        for handler in logging.root.handlers[:]:
            logging.roost.removeHandler(handler)
        log_filename = run_identifier + '.log'
        logging.basicConfig(filename='../logs/' + log_filename,
                            level=logging.INFO,
                            format='%(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s')


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-l", "--log-to-file", default=False,
                        action="store_true")
    parser.add_argument("-v", "--verbose", default=False,
                        action="store_true")
    parser.add_argument("-ae", "--autoencoder", default="temporal", type=str)
    parser.add_argument("--save-model", default=False, action="store_true")
    parser.add_argument("--tensorboard", default=False, action="store_true")
    # Encoder Settings
    parser.add_argument("--encoder", default='vanilla_tcn')
    parser.add_argument("--multiscale", default=False, action="store_true")
    parser.add_argument("--causal-encoder", default=False, action="store_true")
    parser.add_argument("--hierarchical", default=False, action="store_true")
    # Data Settings
    parser.add_argument("-hz", default=0, type=int)
    parser.add_argument("-vt", "--viewing-time",
                        help="Cut raw gaze samples to this value (seconds)",
                        default=-1, type=float)
    parser.add_argument("--signal-type", default='pos', type=str,
                        help="'pos' or 'vel'")
    parser.add_argument("--slice-time-windows", default=None, type=str,
                        help="'2s-overlap' or '2s-disjoint'")
    parser.add_argument("--augment", default=False, action="store_true")
    # Training Settings
    parser.add_argument("--loss-type", default='', type=str,
                        help="supervised or none")
    parser.add_argument("--use-validation-set", default=False, action="store_true")
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--rec-loss", default='mse', type=str,
                        help="mse or bce")
    parser.add_argument("-bs", "--batch-size", default=64, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=5e-4, type=float)
    parser.add_argument("--model-pos", default='', type=str)
    parser.add_argument("--model-vel", default='', type=str)
    # Evaluation Settings
    parser.add_argument("--pca-components", default=0, type=int)
    parser.add_argument("--save-tsne-plot", default=False, action="store_true")
    parser.add_argument("-cv", "--cv-folds", default=5, type=int)
    parser.add_argument("--generate", default=False, action="store_true")
    parser.add_argument("--task", default='', type=str,
                        help="task to train supervised CNN on")
    return parser
