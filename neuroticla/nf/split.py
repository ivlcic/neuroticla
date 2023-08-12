import logging

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

logger = logging.getLogger('nf.split')


def args(nrcla_module: str, parser: ArgumentParser) -> None:
    CommonArguments.data_split(parser)


def main(args) -> int:
    logger.debug("main")
    return 0
