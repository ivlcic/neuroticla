import os
import stanza
import obeliks


def get_stanza_tokenizer(args):
    stanza_dir = os.path.join(args.tmp_dir, 'stanza')
    if not os.path.exists(stanza_dir):
        os.makedirs(stanza_dir)
    if not os.path.exists(os.path.join(stanza_dir, args.lang)):
        stanza.download(args.lang, stanza_dir)
    return stanza.Pipeline(args.lang, dir=stanza_dir, processors="tokenize", download_method=2)


def get_obeliks_tokenizer():
    return obeliks.run


def get_reldi_tokenizer(lang: str):
    if lang == 'sl':
        pass
    else:
        pass