import os
import shutil
import logging

import neuroticla.utils.zip

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments
from neuroticla.core.split import DataSplit
from neuroticla.ner.utils import get_all_languages

logger = logging.getLogger('ner.split')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.split(nrcla_module, parser, 'showeffort')
    parser.add_argument(
        'langs', help='Language files to split.', nargs='+',
        choices=get_all_languages()
    )


def main(arg) -> int:
    for lang in arg.langs:
        files = DataSplit.extract(
            os.path.join(arg.data_in_dir, lang + '.zip'),
            arg.password,
            arg.subsets.split(',') if arg.subsets else None,
            arg.data_out_dir
        )
        DataSplit.file_split(
            lang,
            files,
            arg.data_out_dir,
            arg.data_split,
            arg.non_reproducible_shuffle
        )

    return 0
