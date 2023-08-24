from neuroticla.core.args import CommonArguments
from neuroticla.nf.utils import *

from .binrel import train as binrel_train

logger = logging.getLogger('nf.train')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(nrcla_module, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    labels = ','.join(get_all_labels(nrcla_module))
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
    logger.info("Starting label powerset ...")
    return 0
