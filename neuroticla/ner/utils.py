import os

from argparse import ArgumentParser
from typing import List, Dict

from ..core.args import CommonArguments


def get_all_languages():
    return ['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'bg', 'pl', 'ru', 'sk', 'uk']


def add_common_test_train_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--no_misc', help='Remove MISC tag (replace i with "O").', action='store_true', default=False
    )
    parser.add_argument(
        '--pro', help='Enable Product (PRO) tag.', action='store_true', default=False
    )
    parser.add_argument(
        '--evt', help='Enable Event (EVT) tag.', action='store_true', default=False
    )
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )


def compute_model_path(arg) -> str:
    if not os.path.exists(arg.model_name):
        result_path = os.path.join(arg.result_dir, arg.model_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        result_path = arg.model_name
    return result_path


def get_data_paths_prefixes(arg) -> List[str]:
    data_paths = []
    for lng in arg.langs:
        data_paths.append(os.path.join(arg.data_in_dir, lng))
    return data_paths


def replace_ner_tags(args) -> Dict[str, str]:
    del_misc = {}
    if hasattr(args, 'no_misc') and args.no_misc:
        del_misc['B-MISC'] = 'O'
        del_misc['I-MISC'] = 'O'
    if not hasattr(args, 'pro') or not args.pro:
        del_misc['B-PRO'] = 'O'
        del_misc['I-PRO'] = 'O'
    if not hasattr(args, 'evt') or not args.evt:
        del_misc['B-EVT'] = 'O'
        del_misc['I-EVT'] = 'O'
    return del_misc