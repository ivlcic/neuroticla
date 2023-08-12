import logging

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

logger = logging.getLogger('nf.prep')


def args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(nrcla_module, parser, ('-i', '--data_in_dir'))
    CommonArguments.processed_data_dir(nrcla_module, parser, ('-o', '--data_out_dir'))
    parser.add_argument(
        'input_file',
        help='Corpora file (default: %(default)s)',
        type=str,
        default=''
    )


def main(args) -> int:
    logger.debug("main")
    return 0
