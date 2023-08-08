import logging

from argparse import ArgumentParser
from neuroticla.args import CommonArguments

logger = logging.getLogger('ner.prep')


def args(package: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(package, parser, ('-i', '--data_in_dir'))
    CommonArguments.processed_data_dir(package, parser, ('-o', '--data_out_dir'))
    parser.add_argument('lang', help='language of the text',
                        choices=['sl', 'hr', 'sr', 'bs', 'mk', 'sq', 'cs', 'bg', 'pl', 'ru', 'sk', 'uk'])


def main(args) -> int:
    logger.debug("main")
    return 0
