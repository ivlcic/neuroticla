import logging
import sys
import neuroticla.ner.prep.tokens

from argparse import ArgumentParser
from neuroticla.core.args import CommonArguments

logger = logging.getLogger('ner.test')


def args(nrcla_module: str, parser: ArgumentParser) -> None:
    pass


def test_tokenizer(args) -> int:
    logger.debug("main")
    args.lang = 'sl'
    tokenizer = ner.prep.tokens.get_obeliks_tokenizer(args)
    o = tokenizer('Pozdravljen, svet!\n\nAli pa tudi ne. Kaj pa vem?')
    # print(o.to_conll())
    args.lang = 'hr'
    tokenizer = ner.prep.tokens.get_reldi_tokenizer(args)
    o = tokenizer('Dobrodošao, svijet!\n\nMožda ne. Šta ja znam?')
    # print(o.to_conll())
    return 0
