import os
import logging

from argparse import ArgumentParser
from ..core.args import CommonArguments
from ..core.split import DataSplit

logger = logging.getLogger('nf.split')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.split(module_name, parser)
    parser.add_argument(
        'corpora', help='Corpora files to split.', nargs='+',
        choices=['aussda', 'slomcor']
    )


def main(arg) -> int:
    for corpus in arg.corpora:
        files = DataSplit.extract(
            os.path.join(arg.data_in_dir, corpus + '.zip'),
            arg.password,
            arg.subsets.split(',') if arg.subsets else None,
            arg.data_out_dir
        )
        for f in files:
            base_name = os.path.splitext(os.path.basename(f))[0]
            DataSplit.file_split(
                base_name,
                [f],  # individual file split
                arg.data_out_dir,
                arg.data_split,
                arg.non_reproducible_shuffle
            )
    return 0
