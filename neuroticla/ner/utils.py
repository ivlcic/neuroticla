import os

from argparse import ArgumentParser
from typing import List, Dict

from neuroticla.core.args import CommonArguments


def get_all_languages():
    return ['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'bg', 'pl', 'ru', 'sk', 'uk']


def add_common_test_train_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.split_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(nrcla_module, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(nrcla_module, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '-n', '--model_name', help='Target model name.', type=str, default=None
    )
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
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    parser.add_argument(
        'langs', help='Languages to use.', nargs='+',
        choices=get_all_languages()
    )


def compute_model_name(arg) -> None:
    corpora_prefix = ''
    if hasattr(arg, 'langs'):
        corpora_prefix = '_'.join(arg.langs)
    if arg.model_name is None:
        model_name = arg.pretrained_model + '-' + corpora_prefix
        if arg.no_misc:
            model_name += '-nomisc'
        arg.model_name = model_name


def get_data_path_prefix(arg) -> List[str]:
    corpora_prefix = '_'.join(arg.langs)
    data_path = os.path.join(arg.data_in_dir, corpora_prefix)
    return [data_path]


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