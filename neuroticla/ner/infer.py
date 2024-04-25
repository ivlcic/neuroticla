import logging
import string

from ..core.labels import Labeler
from ..core.trans import TokenClassifyModel
from ..ner.prep.tokens import get_tokenizer
from ..ner.utils import *

logger = logging.getLogger('ner.infer')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.device(parser)
    parser.add_argument(
        'model_name', help='Model name or path.', type=str, default=None
    )
    parser.add_argument(
        'lang', help='Language of the text.',
        choices=get_all_languages()
    )
    parser.add_argument(
        'text', help='Text to test.', type=str, default=None
    )


def main(arg) -> int:
    result_path = compute_model_path(arg)

    mc: TokenClassifyModel = TokenClassifyModel(
        result_path,
        Labeler(
            os.path.join(CommonArguments.data_path('ner', 'processed'), 'tags.csv'),
            replace_labels=replace_ner_tags(arg)
        ),
        device=arg.device
    )
    mc.eval()

    word_tokenizer = get_tokenizer(arg)
    doc = word_tokenizer(arg.text)
    for sent_idx, sentence in enumerate(doc.sentences):
        word_list = [v.text for v in sentence.tokens]
        result = mc.infer(word_list)
        sent_text = ''
        prev_token = None
        for i, v in enumerate(sentence.tokens):
            if not hasattr(v, 'ner'):
                setattr(v, 'ner', result[i]['ner'])
            else:
                v.ner = result[i]['ner']
            if v.ner == 'O':
                if prev_token and prev_token.ner and prev_token.ner != 'O':
                    sent_text += ']-{' + prev_token.ner[2:] + '}'
                if sent_text and v.text not in string.punctuation:
                    sent_text += ' '
            elif v.ner.startswith('B-'):
                if prev_token and prev_token.ner and prev_token.ner != 'O':
                    sent_text += ']-{' + prev_token.ner[2:] + '}'
                if sent_text and v.text not in string.punctuation:
                    sent_text += ' '
                sent_text += '['
            else:
                if sent_text and v.text not in string.punctuation:
                    sent_text += ' '
            sent_text += v.text
            prev_token = v

        logger.info('%s', sent_text)
    return 0
