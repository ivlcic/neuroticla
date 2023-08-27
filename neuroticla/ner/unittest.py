import logging

from argparse import ArgumentParser
from .prep.tokens import get_obeliks_tokenizer, get_reldi_tokenizer

logger = logging.getLogger('ner.unittest')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    pass


def unittest_tokenizer(args) -> int:
    logger.debug("main")
    args.lang = 'sl'
    tokenizer = get_obeliks_tokenizer(args)
    o = tokenizer('Pozdravljen, svet!\n\nAli pa tudi ne. Kaj pa vem?')
    logger.info("Got obeliks result: %s", o)
    args.lang = 'hr'
    tokenizer = get_reldi_tokenizer(args)
    o = tokenizer('Dobrodošao, svijet!\n\nMožda ne. Šta ja znam?')
    logger.info("Got reldi result: %s", o)
    return 0
