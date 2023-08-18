import os
import stanza
import obeliks
import reldi_tokeniser

from stanza.models.common.doc import Document
from reldi_tokeniser.tokeniser import ReldiTokeniser


class Tokenizer:

    def __init__(self, lang: str) -> None:
        self.doc = None
        self.conll = ''
        self.lang = lang

    def tokenize(self, text):
        pass

    def to_conll(self):
        return self.conll


class ObeliksTokenizer(Tokenizer):

    def tokenize(self, text):
        raw_text = '\n'.join(text) if isinstance(text, list) else text

        document = []
        metadocument = []

        if raw_text == '' or raw_text.isspace():
            self.doc = Document(document, raw_text, comments=metadocument)
            return self.doc

        for doc in obeliks.run(raw_text, object_output=True):
            for sentence in doc:
                for word in sentence['sentence']:
                    if word['lemma'] == '_':
                        del (word['lemma'])
                    if word['xpos'] == '_':
                        del (word['xpos'])
                    if word['upos'] == '_':
                        del (word['upos'])
                    if word['misc'] == '_':
                        del (word['misc'])
                document.append(sentence['sentence'])
                metadocument.append(sentence['metadata'].strip().split('\n'))

        self.doc = Document(document, raw_text, comments=metadocument)
        self.doc.lang = self.lang
        return self.doc


class ReldiTokenizer(Tokenizer):

    def tokenize(self, text):
        # self.doc = obeliks.run(text, object_output=True)
        self.conll = reldi_tokeniser.run(text, self.lang, conllu=True)

        raw_text = '\n'.join(text) if isinstance(text, list) else text
        list_of_lines = [el + '\n' for el in raw_text.split('\n')]

        document = []
        metadocument = []

        if raw_text == '' or raw_text.isspace():
            self.doc = Document(document, raw_text, comments=metadocument)
            return self.doc

        reldi = ReldiTokeniser(self.lang, conllu=True)
        for doc in reldi.run(list_of_lines, mode='object'):
            for sentence in doc:
                for word in sentence['sentence']:
                    if word['lemma'] == '_':
                        del (word['lemma'])
                    if word['xpos'] == '_':
                        del (word['xpos'])
                    if word['upos'] == '_':
                        del (word['upos'])
                    if word['misc'] == '_':
                        del (word['misc'])
                document.append(sentence['sentence'])
                metadocument.append(sentence['metadata'].strip().split('\n'))
        self.doc = Document(document, raw_text, comments=metadocument)
        self.doc.lang = self.lang
        return self.doc


def get_stanza_tokenizer(args):
    stanza_dir = os.path.join(args.tmp_dir, 'stanza')
    if not os.path.exists(stanza_dir):
        os.makedirs(stanza_dir)
    if not os.path.exists(os.path.join(stanza_dir, args.lang)):
        stanza.download(args.lang, stanza_dir)
    return stanza.Pipeline(args.lang, dir=stanza_dir, processors="tokenize", download_method=2)


def get_obeliks_tokenizer(args):
    tokenizer = ObeliksTokenizer(args.lang)
    return tokenizer.tokenize


def get_reldi_tokenizer(args):
    tokenizer = ReldiTokenizer(args.lang)
    return tokenizer.tokenize


def get_tokenizer(args):
    if args.lang in ['sl']:
        return get_obeliks_tokenizer(args)
    elif args.lang in ['bg', 'hr', 'sr', 'mk']:
        return get_reldi_tokenizer(args)
    else:
        return get_stanza_tokenizer(args)
