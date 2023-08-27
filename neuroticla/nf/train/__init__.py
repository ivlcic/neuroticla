from ...nf.utils import *

from .binrel import train as binrel_train
from .lpset import train as lpset_train

logger = logging.getLogger('nf.train')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(module_name, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    labels = ','.join(get_all_labels(module_name))
    parser.add_argument(
        '-u', '--subset', type=str, default=None,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])

    parser.add_argument('corpora', type=str, default=None,
                        help='Corpora prefix or path prefix to use for training')


def train_binrel(arg) -> int:
    logger.info('Starting binary relevance training ...')
    binrel_train(arg)
    return 0


def train_lpset(arg) -> int:
    logger.info("Starting label power-set ...")
    lpset_train(arg)
    return 0
