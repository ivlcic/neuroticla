import logging
import os
from argparse import ArgumentParser
from typing import List

from ..core.args import CommonArguments

logger = logging.getLogger('nf.utils')


def get_all_labels(module_name: str) -> List[str]:
    labels_file = os.path.join(CommonArguments.data_path(module_name, 'processed'), 'tags.csv')
    labels = open(labels_file, 'r').read().split('\n')
    return labels


def get_labels(module_name: str, arg) -> List[str]:
    labels = get_all_labels(module_name)
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


def compute_model_name_old(arg, labels: List[str] = None, force_label: bool = False) -> str:
    l_str = ''
    if labels is not None:
        l_str = '-' + '_'.join(labels)
    if arg.model_name is not None:
        if force_label:
            m = f'{arg.model_name}{l_str}'
        else:
            m = arg.model_name
    else:
        m = f'{arg.pretrained_model}.e{arg.epochs}.b{arg.batch}.l{arg.learn_rate}-{arg.corpora}{l_str}'
    return m


def compute_model_name(arg, text_fields: List[str], labels: List[str] = None, force_label: bool = False) -> str:
    l_str = ''
    if labels is not None:
        l_str = '.l-' + '_'.join(labels)
    f_str = ''
    if text_fields is not None:
        f_str = '.f-' + '_'.join([text_field[0] for text_field in text_fields if text_field])
    if arg.model_name is not None:
        if force_label:
            m = f'{arg.model_name}{f_str}{l_str}'
        else:
            m = arg.model_name
    else:
        m = f'{arg.pretrained_model}.e{arg.epochs}.b{arg.batch}.l{arg.learn_rate}.{arg.corpora}{f_str}{l_str}'
    return m


def compute_model_path(arg, subdir: str) -> str:
    if not os.path.exists(arg.model_name):
        result_path = os.path.join(arg.result_dir, subdir, arg.model_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        result_path = arg.model_name
    return result_path


def add_common_test_train_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--max_seq_len', help='Max sub-word tokens length.', type=int, default=512
    )
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
