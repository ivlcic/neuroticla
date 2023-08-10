import logging

import ner.prep.tokens

from argparse import ArgumentParser
from neuroticla.args import CommonArguments

logger = logging.getLogger('ner.test')


def args(package: str, parser: ArgumentParser) -> None:
    pass


def main(args) -> int:
    logger.debug("main")
    tokenizer = ner.prep.tokens.get_obeliks_tokenizer()
    o = tokenizer('Pozdravljen, svet!', object_output=True)
    return 0