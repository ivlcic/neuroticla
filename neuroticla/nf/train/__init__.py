import json
import logging

from transformers import TrainingArguments

from neuroticla.core.dataset import TokenClassifyDataset
from neuroticla.core.json import NpEncoder
from neuroticla.core.labels import Labeler
from neuroticla.core.split import DataSplit
from neuroticla.core.trans import TokenClassifyModel, ModelContainer
from neuroticla.ner.utils import *

logger = logging.getLogger('nf.train')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    add_common_test_train_args(nrcla_module, parser)
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])


def train_binrel(arg) -> 0:
    logger.info("Starting train_binrel...")
    return 0


def train_lpset(arg) -> 0:
    logger.info("Starting train_binrel...")
    return 0
