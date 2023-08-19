import logging

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

logger = logging.getLogger('nf.split')


def add_args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.processed_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.split_data_dir(nrcla_module, parser, ('-o', '--data_out_dir'))
    CommonArguments.tmp_dir(nrcla_module, parser, ('-t', '--tmp_dir'))
    CommonArguments.data_split(parser)
    parser.add_argument(
        '-u', '--subsets', type=str, default=None,
        help="Subsets of the files to use for each corpora (file name contains any of the comma separated strings)",
    )
    parser.add_argument(
        '-p', '--password', type=str, required=True, help="Zip file password"
    )
    parser.add_argument(
        '-r', '--non_reproducible_shuffle', action='store_true', default=False,
        help='Non reproducible data shuffle.',
    )
    parser.add_argument(
        'corpora', help='Corpora files to split.', nargs='+',
        choices=['aussda', 'slomcor']
    )


def main(args) -> int:
    logger.debug("main")
    return 0
