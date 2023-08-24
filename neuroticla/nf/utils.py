import os
import logging

from argparse import ArgumentParser
from typing import List, Dict

from neuroticla.core.args import CommonArguments

logger = logging.getLogger('nf.utils')


def get_all_labels(ncra_module: str) -> List[str]:
    labels_file = os.path.join(CommonArguments.data_path(ncra_module, 'processed'), 'tags.csv')
    labels = open(labels_file, 'r').read().split('\n')
    return labels


def get_labels(nrcla_module: str, arg) -> List[str]:
    labels = get_all_labels(nrcla_module)
    if arg.subset is not None:
        subset = arg.subset.split(',')
        if all(item in labels for item in subset):
            labels = subset
        else:
            logger.error('Invalid labels subset: %s', arg.subset)
            return []
    return labels


def get_data_path_prefix(arg) -> List[str]:
    data_paths = [os.path.join(arg.data_in_dir, arg.corpora)]
    return data_paths


def compute_model_path(arg) -> str:
    if not os.path.exists(arg.model_name):
        result_path = os.path.join(arg.result_dir, arg.model_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        result_path = arg.model_name
    return result_path


def add_common_test_train_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(nrcla_module, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(nrcla_module, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
