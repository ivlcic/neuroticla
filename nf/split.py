import logging

from argparse import ArgumentParser
from neuroticla.args import CommonArguments

logger = logging.getLogger('nf.split')


def args(package: str, parser: ArgumentParser) -> None:
    CommonArguments.data_split(parser)


def main(args) -> int:
    logger.debug("main")
    return 0
